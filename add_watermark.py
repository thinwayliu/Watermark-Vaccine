import numpy as np

def add_watermark_to_image(image,xs, watermark,sl):
    rgba_image = image.convert('RGBA')
    rgba_watermark = watermark.convert('RGBA')

    image_x, image_y = rgba_image.size
    watermark_x, watermark_y = rgba_watermark.size

    # 缩放图片
    scale = sl
    watermark_scale = min(image_x / (scale * watermark_x), image_y / (scale * watermark_y))
    new_size = (int(watermark_x * watermark_scale), int(watermark_y * watermark_scale))
    #rgba_watermark = rgba_watermark.resize(new_size)
    rgba_watermark = rgba_watermark.resize(new_size, resample=Image.ANTIALIAS)
    # 透明度
    rgba_watermark_mask = rgba_watermark.convert("L").point(lambda x: min(x,int(xs[0])))
    rgba_watermark.putalpha(rgba_watermark_mask)

    watermark_x, watermark_y = rgba_watermark.size
    # 水印位置
    #rgba_image.paste(rgba_watermark, (0, 0), rgba_watermark_mask) #右下角
    ##限制水印位置

    a=np.array(xs[1])
    a=np.clip(a, 0, 224-watermark_x)

    b= np.array(xs[1])
    b = np.clip(b, 0, 224 - watermark_y)




    x_pos=int(a)
    y_pos=int(b)
    rgba_image.paste(rgba_watermark, (x_pos, y_pos), rgba_watermark_mask)  # 右上角
    rgba_watermark.save('newlogo.png')
    return rgba_image