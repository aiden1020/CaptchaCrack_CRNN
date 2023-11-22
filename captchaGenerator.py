from PIL import Image, ImageDraw, ImageFont
import random
import os
import csv


def captchaGen(labelCSV, imgOutput_path, imgNum):
    directory = imgOutput_path
    os.makedirs(directory, exist_ok=True)

    # make label csv
    with open(labelCSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['path', 'label'])

        for i in range(imgNum):
            width, height = 120, 36  # get width and height by target.png

            # get gray background randomly
            gray_intensity = random.randint(200, 220)
            background_color = (gray_intensity, gray_intensity, gray_intensity)

            # create image
            image = Image.new("RGB", (width, height), background_color)
            draw = ImageDraw.Draw(image)

            # create random nosie for background
            for y in range(height):
                for x in range(width):
                    noise = random.randint(0, 30)
                    pixel_color = (
                        background_color[0] - noise, background_color[1] - noise, background_color[2] - noise)
                    draw.point((x, y), fill=pixel_color)

            # create 4 digit random captcha
            captcha = '   '.join(random.choices('0123456789', k=4))

            # random offset for placing captcha
            x_offset = random.randint(5, 8)
            y_offset = random.randint(1, 4)

            for char in captcha:
                font_path = "/System/Library/Fonts/HelveticaNeue.ttc"
                font_size = random.randint(29, 31)  # random font size
                font = ImageFont.truetype(font_path, font_size)

                # random text color
                text_color = (random.randint(0, 255), random.randint(
                    0, 255), random.randint(0, 255))

                # random bold font
                stroke_width = random.randint(0, 1)

                draw.text((x_offset, y_offset), char, font=font,
                          fill=text_color, stroke_width=stroke_width)
                x_offset += font.font.getsize(char)[0][1]+2

            # create random nosie lines
            for _ in range(random.randint(4, 6)):
                line_color = tuple(random.randint(0, 255) for _ in range(3))
                x1 = random.randint(0, width)
                y1 = random.randint(0, height)
                x2 = random.randint(0, width)
                y2 = random.randint(0, height)
                draw.line((x1, y1, x2, y2), fill=line_color)
            path = "{}/{}.png".format(directory, i+1)
            imgName = "{}.png".format(i+1)
            label = captcha.replace(' ', '')
            image.save(path, 'PNG')
            writer.writerow([imgName, label])


captchaGen(labelCSV='./trainingSet/label.csv',
           imgOutput_path='./trainingSet', imgNum=1000)
