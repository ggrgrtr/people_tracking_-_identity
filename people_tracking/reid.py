import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# используем пре-трэин модель реснет50
from torchvision import models

from .utils import resolve_reid_weights


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


class GeMPooling2d(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = float(p)
        self.eps = float(eps)

    # GeM pooling 
    # лучше сохраняет заметные  признаки одежды, текстуры, силуэта
    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        # L2 нормализация после GeM pooling, чтобы сохранить масштаб признаков
        #  и обеспечить стабильность при сравнении дескрипторов
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        return x.pow(1.0 / self.p)

# извлечение признаков персоны (опорная модель)
# берем из конфига статичные свойства
class ResNet50ReIDBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_parts = max(2, int(config.reid_num_parts))
        # коэффициент для глобального и частичного признаков,
        #  который определяет их относительное влияние на итоговый дескриптор,
        #  что позволяет адаптировать модель к условиям
        self.global_weight = float(config.reid_global_weight)
        self.part_weight = float(config.reid_part_weight)
        # сверточный слой с обобщенным средним пуллингом
        self.gem_pool = GeMPooling2d(config.reid_gem_p)
        # адаптивный пуллинг для выделения частей тела, чтобы лучше потом различать одежду и цвета
        self.part_pool = nn.AdaptiveAvgPool2d((self.num_parts, 1))

        # берем реснет50 с ImageNet-весами (4 н.слоями и ф.акт, пулингм и тд)
        try:
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.pretrained_source = "ImageNet"
        except Exception as exc:
            print(f"Warning!!!!: failed to load ImageNet weights for ReID backbone: {exc}")
            backbone = models.resnet50(weights=None)
            self.pretrained_source = "random_init"
        
        # контейнер объединения нейронных слоев
        # сверточные слои реснет50 до глобального среднего пуллинга, который мы заменим на свой глобальный GeM 
        # выносим: конфолюция + батч-норм + релу + макспуллинг
        
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool, 
        )
        # перенос слоев из реснет50 в наш класс бэкбон
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.feature_dim = 2048 * (1 + self.num_parts)

    # итерация прохода данных по слоям нейронной сети
    def forward(self, x):
        x = self.stem(x) # начальный блок обработки, который извлекает базовые признаки
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # дескриптор персоны
        # flatten - схлопываем изо. к [1, 2048]
        global_feat = self.gem_pool(x).flatten(1)
        # финальный эмбендинг 
        global_feat = F.normalize(global_feat, p=2, dim=1)

        
        part_feat = self.part_pool(x).squeeze(-1).transpose(1, 2)
        part_feat = F.normalize(part_feat, p=2, dim=2)
        part_feat = part_feat.reshape(part_feat.size(0), -1)

        # комбинируем глобальный и частичный признаки с учетом их весов
        # чтобы получить более устойчивый и информативный дескриптор для идентификации
        descriptor = torch.cat(
            [
                global_feat * np.sqrt(max(1e-6, self.global_weight)),
                part_feat * np.sqrt(max(1e-6, self.part_weight)),
            ],
            dim=1,
        )
        return F.normalize(descriptor, p=2, dim=1)

# нормализация вектора признаков, создание единичной длины,
def normalize_feature(vector):
    if vector is None:
        return None

    # преобразуем вектор в массив numpy с типом float32 для оптимальной работы с нейронными сетями и вычислениями
    feature = np.asarray(vector, dtype=np.float32)
    # нормируем вектор признаков, чтобы его длина была равна 1, что позволяет использовать косинусное сходство для сравнения признаков
    norm = np.linalg.norm(feature)
    # делим вектор на его норму, если норма достаточно велика, чтобы избежать деления на ноль, и возвращаем нормализованный вектор признаков
    if norm > 1e-8:
        feature = feature / norm
    return feature.astype(np.float32)

# косинусное сходство между двумя векторами признаков, 
# которое используется для оценки сходства между треком и детекцией
def cosine_similarity(feature_a, feature_b):
    if feature_a is None or feature_b is None:
        return -1.0

    a = normalize_feature(feature_a)
    b = normalize_feature(feature_b)
    # вычисляем косинусное сходство как скалярное произведение нормализованных
    #  векторов признаков, что дает значение от -1 до 1,  1 - полное совпадение
    return float(np.dot(a, b))

# сравнение цветовых гистограмм
def color_similarity(hist_a, hist_b):
    if hist_a is None or hist_b is None:
        return -1.0

    # пересечение гистограмм, которая измеряет общую площадь пересечения между двумя гистограммами,
    #  что дает представление о том, насколько похожи цвета в двух образцах
    intersection = float(np.minimum(hist_a, hist_b).sum())

    # L1 расстояние (Манхетенское) - модуль разн коорд
    # преобразуем его в сходство, вычитая нормированное расстояние из 1,
    # что дает более высокое значение для более похожих гистограмм
    l1_similarity = max(0.0, 1.0 - 0.5 * float(np.abs(hist_a - hist_b).sum()))
    # комбинируем оба метода с весами, чтобы получить более надежную оценку сходства цветов,
    #  учитывая общую структуру гистограмм и их конкретные различия
    
    # intersection хорошо работает для цветовых гистограмм
    #  и лучше отражает общие совпадающие цвета
    # l1_similarity добавляет общий штраф за расхождение всей формы гистограммы

    # основной критерий: есть ли у двух людей похожее распределение цветов одежды
    # дополнительная поправка: насколько сильно вся гистограмма в целом отличается
    return 0.7 * intersection + 0.3 * l1_similarity

# смешивание старого и нового признаков с учетом момента, который определяет,
#  насколько сильно новый признак должен влиять на итоговый результат,
#  что позволяет адаптироваться к изменениям внешности объекта,
# сохраняя при этом устойчивость к временным изменениям
def blend_feature(old_feature, new_feature, momentum=0.88):
    if old_feature is None:
        return normalize_feature(new_feature)
    if new_feature is None:
        return old_feature

    feature = momentum * old_feature + (1.0 - momentum) * new_feature
    return normalize_feature(feature)

# смешивание старой и новой цветовой гистограммы с учетом момента, который определяет,
#  насколько сильно новая гистограмма должна влиять на итоговый результат
def blend_histogram(old_hist, new_hist, momentum=0.82):
    if old_hist is None:
        return new_hist
    if new_hist is None:
        return old_hist

    hist = momentum * old_hist + (1.0 - momentum) * new_hist
    hist_sum = float(hist.sum())
    if hist_sum > 1e-8:
        hist = hist / hist_sum

    return hist.astype(np.float32)


# по bbox через кроп создает глубокие признаки персоны и цветовые гистограммы
class AppearanceEncoder:
    def __init__(self, config, device, base_dir):
        # config - все настройки ReID
        self.config = config
        self.device = device
        # размер кропа, который определяется на основе настроек конфигурации ReID
        self.input_size = (int(config.reid_input_width), int(config.reid_input_height))
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        # создаем модель ReID и загружаем специализированные веса
        self.model, self.weights_path = self._build_model(device, base_dir)
        # описание модели, включая информацию о том, какие веса используются
        if self.weights_path is not None:
            self.description = (
                f"{config.reid_backbone} with specialized weights "
                f"({self.weights_path.name})"
            )
        else:
            self.description = (
                f"{config.reid_backbone} with ImageNet-only fallback "
                "(add ReID weights for best ID stability)"
            )

    # извлекает словарь весов из словаря состояния чекпоинта
    def _extract_state_dict(self, checkpoint):
        if isinstance(checkpoint, dict):
            # попытка найти словарь весов в различных общих ключах,
            # которые могут использоваться в разных форматах чекпоинтов, чтобы обеспечить совместимость с различными источниками весов
            for key in ("state_dict", "model", "net"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    return checkpoint[key]
        # возвращаем словарь состояния, если он уже является объектом класса словарь, иначе возвращаем None, что формат не поддерживается
        # Если  нет "state_dict", "model", "net", то сам checkpoint уже словарь весов, он возвращает его напрямую
        return checkpoint if isinstance(checkpoint, dict) else None


    # приводим словарь весов к формату реснет50, удаляя возможные префиксы и адаптируя ключи для начального блока модели,
    # чтобы обеспечить успешную загрузку весов в модель, даже если исходный чекпоинт использует другую нотацию для этих слоев
    def _clean_state_dict(self, state_dict):
        cleaned = {}
        for key, value in state_dict.items():
            new_key = str(key)
            # убираем загруженные префиксы, которые могут быть добавлены при сохранении модели в разных форматах
            for prefix in ("module.", "model.", "backbone.", "base."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]

            # адаптируем ключи для начального блока модели, чтобы они соответствовали структуре нашего бэкбона, что позволяет успешно загрузить веса в модель, даже если исходный чекпоинт использует другую нотацию для этих слоев
            # backbone.conv1 - stem.0,   backbone.bn1 - stem.1 и тд
            if new_key.startswith("conv1."):
                new_key = "stem.0." + new_key[len("conv1.") :]
            elif new_key.startswith("bn1."):
                new_key = "stem.1." + new_key[len("bn1.") :]

            cleaned[new_key] = value
        return cleaned

    # ВЕСа В МОДЕЛЬ
    # загружает специализированные веса в модель, обрабатывая различные форматы чекпоинтов из файла .pth
    # и обеспечивая совместимость с архитектурой модели
    def _load_specialized_weights(self, model, weights_path):
        # берем веса weights_path из скаченного файла типа Реид.pth в репозиторий
        try:
            # checkpoint это словарь состояния нейронки. он содержит веса, оптимизатор, эпохи и т.д
            checkpoint = torch.load(
                str(weights_path),
                map_location="cpu",
                weights_only=False,
                encoding="latin1",
            )
        
        except Exception as exc:
            print(f"Warning: failed to load ReID checkpoint {weights_path}: {exc}")
            return False

        state_dict = self._extract_state_dict(checkpoint)
        # state_dict это уже словарь параметров нейросети после извлечения 
        # из всей стркутуры модели в виде словаря
        if state_dict is None:
            # если не удалось извлечь словарь весов, значит формат чекпоинта не поддерживается
            print(f"Warning: unsupported ReID checkpoint format: {weights_path}")
            return False


        cleaned_state = self._clean_state_dict(state_dict)
        try:
            # загружаем веса только совпавших слоев - strict=False и возвращаем предупреждения
            # загрузка новых параметров в уже созданные слои реснет50 для лучшего распознавания

            # слой stem.0 это всё тот же Conv2d
            # weight для сллоя Реснет50 берётся уже из ReID checkpoint, если ключ совпал, и применяется к модели, которая изначально была инициализирована с ImageNet-весами, что позволяет адаптировать модель к задаче ReID, используя специализированные веса, которые были обучены на задачах идентификации людей, и обеспечивая более стабильные и точные признаки для отслеживания.

            # параметры backbone после загрузки становятся не просто ImageNet,
            #  а ближе к задаче re-identification
            missing_keys, unexpected_keys = model.load_state_dict(cleaned_state, strict=False)
        except Exception as exc:
            print(f"Warning: failed to apply ReID weights {weights_path}: {exc}")
            return False

        # проверка ошибок
        if missing_keys:
            print(
                "Warning: ReID checkpoint loaded with missing keys: "
                + ", ".join(missing_keys[:6])
                + (" ..." if len(missing_keys) > 6 else "")
            )
        if unexpected_keys:
            print(
                "Warning: ReID checkpoint has unexpected keys: "
                + ", ".join(unexpected_keys[:6])
                + (" ..." if len(unexpected_keys) > 6 else "")
            )
        return True

    def _build_model(self, device, base_dir):
        # создаем объект с resnet50
        model = ResNet50ReIDBackbone(self.config)
        # возвращаем веса для РеиД
        weights_path = resolve_reid_weights(base_dir, self.config.reid_weights)
        if weights_path is not None:
            # загружает более подходящие ReID-веса
            loaded = self._load_specialized_weights(model, weights_path)
            if not loaded:
                weights_path = None

        # меняем режим модели на валидационынй - замораживаем веса
        model.eval().to(device)
        return model, weights_path

    def _crop_person(self, frame, bbox, pad=None):
        frame_h, frame_w = frame.shape[:2]
        x, y, w, h = [int(v) for v in bbox]
        pad = self.config.reid_padding if pad is None else pad

        pad_x = int(w * pad)
        pad_y = int(h * pad)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame_w, x + w + pad_x)
        y2 = min(frame_h, y + h + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def _compute_histogram(self, hsv_crop):
        # считает гистограмму по оттенку и насыщенности
        # 12*8 бинов в reid_hist_h_bins  и reid_hist_s_bins
        hist = cv2.calcHist(
            [hsv_crop],
            [0, 1],
            None,
            [self.config.reid_hist_h_bins, self.config.reid_hist_s_bins],
            [0, 180, 0, 256],
        )
        hist = hist.astype(np.float32)
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
        # возвращаем паддинг гистограммы вытянутого в одномерныйй массив
        return hist.flatten()

    def _build_color_descriptor(self, crop):
        # BGR в HSV
        # H отвечает за оттенок, S за насыщенность, V за яркость
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h = hsv.shape[0]

        # делим кроп на секции
        # зоны специально пересекабтся, чтобы оценка цвета была более устойчивой к небольшим изменениям позы или кропа
        upper_start = int(h * 0.05)
        upper_end = min(h, max(upper_start + 8, int(h * 0.58)))
        lower_start = min(h - 1, max(0, int(h * 0.38)))
        lower_end = min(h, max(lower_start + 8, int(h * 0.98)))

        sections = [
            hsv,
            hsv[upper_start:upper_end, :],
            hsv[lower_start:lower_end, :],
        ]
        # Цвет описывается отдельно по всему силуэту, верху и внизу, чтобы лучше различать одежду
        descriptors = []

        for section in sections:
            if section.size == 0:
                section = hsv

            hist = self._compute_histogram(section)
            # нормализуем цвета от 0 до 1:
            # средняя насыщенность зоны
            # средняя яркость зоны
            mean_sat = float(section[..., 1].mean()) / 255.0
            mean_val = float(section[..., 2].mean()) / 255.0
            # добавляем к дескриптору гистограммы 2 бина средней насыщенности и яркости
            descriptors.append(np.concatenate([hist, np.array([mean_sat, mean_val], dtype=np.float32)]))

        # склеиваем дескрипторы всех секций в один вектор, который описывает цветовую характеристику персоны
        #  учитывая общую цветовую схему и особенности верхней и нижней части одежды,
        #  что позволяет более точно различать людей по цвету их одежды
        descriptor = np.concatenate(descriptors).astype(np.float32)

        # Нормализуем итоговый дескриптор, чтобы его значения были в диапазоне от 0 до 1, обеспечивает стабильность при сравнении цветовых дескрипторов между треками и детекциями
        descriptor_sum = float(descriptor.sum())
        if descriptor_sum > 1e-8:
            descriptor = descriptor / descriptor_sum
        # возвращаем np.float32 по трем зонам: целиком, верх, низ

        # если глубокий ReID-признак не посчитан или он ненадёжен,
        #  цветовой дескриптор всё равно помогает понять
        #  похож ли новый bbox на уже известный трек
        return descriptor

    # к 128x256
    def _preprocess_crop(self, crop):
        resized = cv2.resize(crop, self.input_size, interpolation=cv2.INTER_LINEAR)
        # меняем цветовую схему на RGB, тк. OpenCV по умолчанию использует BGR
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # преобразуем в тензор и нормализуем перестановой эл. в тензоре,
        #  чтобы привести к виду, который использует нейронка
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()
        tensor = tensor.div_(255.0)
        tensor = tensor.sub_(self.norm_mean).div_(self.norm_std)
        return tensor

    # останавливаем град.спуск для оптимизации и неизменения весов модели
    #  при извлечении признаков, так как мы не обучаем модель,
    #  а просто используем ее для получения дескрипторов
    @torch.no_grad()

    # вырезается кроп с падингом
    # выявляеются цвета кропа в цветовых дескрипторах
    def extract(
        self,
        frame,
        boxes,
        include_features=True,
        max_feature_boxes=None,
        feature_indices=None,
    ):
        features = [None] * len(boxes)
        color_histograms = [None] * len(boxes)
        if not boxes:
            return features, color_histograms

        # определяем, для каких боксов извлекать глубокие признаки, основываясь на параметрах include_features, max_feature_boxes и feature_indices из конфигурации ReID, чтобы оптимизировать производительность, извлекая признаки только для наиболее релевантных боксов, таких как те, которые имеют больший размер или находятся в определенных позициях
        selected_feature_indices = set()
        if include_features:
            if feature_indices is not None:
                selected_feature_indices = {
                    int(index)
                    for index in feature_indices
                    if 0 <= int(index) < len(boxes)
                }
            else:
                ordered_indices = sorted(
                    range(len(boxes)),
                    key=lambda index: max(1, int(boxes[index][2])) * max(1, int(boxes[index][3])),
                    reverse=True,
                )
                if max_feature_boxes is not None and max_feature_boxes > 0:
                    ordered_indices = ordered_indices[: max_feature_boxes]
                selected_feature_indices = set(ordered_indices)

        tensors = []
        valid_indices = []

        for index, bbox in enumerate(boxes):
            crop = self._crop_person(frame, bbox)
            if crop is None:
                continue

            # Цветовой дескриптор считаем почти всегда. он дешевле и помогает даже без глубокого ReID
            color_histograms[index] = self._build_color_descriptor(crop)
            if include_features and index in selected_feature_indices:
                tensors.append(self._preprocess_crop(crop))
                valid_indices.append(index)

        if not tensors:
            return features, color_histograms

        batch = torch.stack(tensors).to(self.device)
        # model(batch) - получаем embedding
        # detach() - отсоединяем от графа вычислений, чтобы не сохранять градиенты и оптимизировать память,
        #  так как мы не обучаем модель, а просто извлекаем признаки
        embeddings = self.model(batch).detach().cpu().numpy().astype(np.float32)

        if embeddings.ndim == 1:
            embeddings = embeddings[None, :]

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-8)

        for index, embedding in zip(valid_indices, embeddings):
            features[index] = normalize_feature(embedding)

        return features, color_histograms

