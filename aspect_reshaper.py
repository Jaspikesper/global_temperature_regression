from PIL import Image


def stretch_and_crop_image(input_file, output_file, target_aspect_ratio):
    # Open the input image
    img = Image.open(input_file)
    original_width, original_height = img.size
    print(f"Original image size: {original_width}x{original_height}")

    # Calculate new dimensions based on the target aspect ratio (9/16)
    # Here, we keep the width and compute new height
    new_width = original_width
    new_height = round(new_width / target_aspect_ratio)  # new_height = width / (9/16) = width * (16/9)
    print(f"Resized image dimensions (for {target_aspect_ratio} aspect ratio): {new_width}x{new_height}")

    # Stretch the image to the new dimensions using a high-quality resampling filter
    resized_img = img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)

    # Crop the image to a square by clipping the top and bottom.
    # The square will have a size equal to the original width.
    crop_top = (new_height - new_width) // 2
    crop_bottom = crop_top + new_width  # This defines the bottom boundary for a centered square crop
    cropped_img = resized_img.crop((0, crop_top, new_width, crop_bottom))
    print(f"Cropped square image dimensions: {cropped_img.size}")

    # Save the final cropped image
    cropped_img.save(output_file)
    print(f"Final image saved as {output_file}")


if __name__ == '__main__':
    input_file = 'background.png'
    output_file = 'example2.png'
    target_aspect_ratio = 9 / 16  # Aspect ratio: width/height = 9/16
    stretch_and_crop_image(input_file, output_file, target_aspect_ratio)
