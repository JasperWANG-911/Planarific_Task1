import fitz
from PIL import Image
import os

Image.MAX_IMAGE_PIXELS = None

# Convert PDF to PNG
def convert_pdf_to_image(pdf_path, dpi):
    pdf_document = fitz.open(pdf_path)
    page = pdf_document.load_page(0)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    image_path = "cropped_image.png"
    pix.save(image_path)

# Crop the image
def crop_image(image_path, square_size, output_folder):
    image = Image.open(image_path)
    width, height = image.size
    x_steps = (width + square_size - 1) // square_size
    y_steps = (height + square_size - 1) // square_size

    if not os.path.exists(output_folder):  # Create the output folder if it doesn't exist
        os.makedirs(output_folder)

    count = 0
    for y in range(y_steps):
        for x in range(x_steps):
            left = x * square_size
            upper = y * square_size
            right = min(left + square_size, width)
            lower = min(upper + square_size, height)

            # Crop and save the square
            cropped_image = image.crop((left, upper, right, lower))
            output_path = os.path.join(output_folder, f'crop_{count}.png')
            cropped_image.save(output_path)
            count += 1
            print(f'{count}_done')


# edit pdf_path
pdf_path = "68484 - Longley Area Park, Sheffield - Site 3 - Orthomosaic.pdf"

# edit parameters
convert_pdf_to_image(pdf_path, dpi=1200)
crop_image('cropped_image.png', 2700, 'cropped_images')


