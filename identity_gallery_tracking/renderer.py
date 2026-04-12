import cv2
from .utils import track_color

# визуализация анализа изображения: отрисовка пути трека, айди, рамки

# берем конкретный объект tracklet и отрисовываем полученнуб из него информацию
def draw_tracklet(frame, tracklet):
    #  сглаженной рамки
    x, y, w, h = [int(v) for v in tracklet.smooth_bbox]
    # цвет надписи айди и рамки
    color_key = tracklet.person_id if tracklet.person_id is not None else tracklet.id
    color = track_color(color_key)


    # если человеку уже присвоен устойчивый identity, пишет ID person_id
    # если identity еще нет, пишет Track id
    label = f"ID {tracklet.person_id}" if tracklet.person_id is not None else f"Track {tracklet.id}"

    # если tracklet уже подтвержден, но identity еще не присвоена, добавляет pending
    if tracklet.person_id is None and tracklet.is_confirmed(tracklet.min_confirmed_hits):
        label += " pending"
    
    # если tracklet еще совсем новый и не подтвержден, добавляем троеточие
    elif not tracklet.is_confirmed(tracklet.min_confirmed_hits):
        label += " ..."

    # рамка
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    # айди
    cv2.putText(
        frame,
        label,
        (x, max(24, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        color,
        2,
    )

    
    # Маршрут рисуем как полилинию, а последним сегментом добавляем стрелку,
    # чтобы на статичном кадре была видна не только траектория, но и направление движения.

    # получаем из треклет deque с элементами из координат, превращаем его в список
    # tracklet.path -  история точек перемещения человека, центр bbox
    points = list(tracklet.path)

    for index in range(1, len(points)):
        # рисуем линию между парами соседних точек траектории с цветом трека
        cv2.line(frame, points[index - 1], points[index], color, 2)
    # добавляю стрелочку в конце
    if len(points) >= 2:
        cv2.arrowedLine(frame, points[-2], points[-1], color, 2, tipLength=0.25)


# отрисовка ФПс, запсиь, если есть, кол-во треков и опред. людей
def draw_dashboard(frame, source_fps, active_tracks, identified_tracks, recording):
    cv2.putText(
        frame,
        f"FPS: {source_fps:.1f}",
        (frame.shape[1] - 190, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Tracklets: {active_tracks}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Identified: {identified_tracks}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    if recording:
        cv2.putText(
            frame,
            "REC",
            (frame.shape[1] - 90, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
