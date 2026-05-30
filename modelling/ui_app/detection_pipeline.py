def detect_accident_from_result(result) -> bool:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return False

    names = getattr(result, "names", {}) or {}
    for cls_idx in boxes.cls:
        class_id = int(cls_idx)
        class_name = ""
        try:
            if isinstance(names, dict):
                class_name = str(names.get(class_id, ""))
            else:
                class_name = str(names[class_id])
        except Exception:
            class_name = ""

        if "accident" in class_name.lower():
            return True

    for cls_idx in boxes.cls:
        if int(cls_idx) == 0:
            return True

    return len(boxes) > 0

def detect_accident_from_collection(results) -> bool:
    if results is None:
        return False
    return any(detect_accident_from_result(result) for result in results)
