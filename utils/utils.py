def meters_to_pixels(meters, scale):
    """将米转换为像素"""
    return meters * scale

def pixels_to_meters(pixels, scale):
    """将像素转换为米"""
    return pixels / scale

def create_rect_from_center(center_x, center_y, width, height, scale):
    """创建基于中心的矩形"""
    return (
        (center_x - width/2) * scale,
        (center_y - height/2) * scale,
        width * scale,
        height * scale
    )