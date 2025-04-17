# Project Directory Overview

Welcome to the **Math Club's Climate Change Project**! This document provides a guide for and modifying images to work effectively with our interactive regression plot in `main.py`.

## Image Preparation for `main.py`

To prepare a photo for use in the regression plot (`main.py`), you need to reshape it to match the required aspect ratio (9:16). Use the provided `aspect_reshaper.py` script as follows:

### Step-by-Step Guide:

1. **Place your original image** in the `static` folder.
2. **Navigate to the project root** and execute the reshaper script:

   ```
   python aspect_reshaper.py
   ```

   Modify the ouputted filename in `aspect_reshaper.py` by changing it to useMe.png

   ```
   input_file = 'your_image.png'       # Original image
   output_file = 'reshaped_image.png'  # Output image
   target_aspect_ratio = 9 / 16        # Required aspect ratio
   ```
3. **Move** the reshaped image into the `static` folder for usage.


## General Project Information

For textual notes including the rubric, please refer to the documents located in the:

* `project_high-levels` folder

This contains important contextual information and high-level documentation relevant to the entire team's workflow.

---

**Note:** Always verify your image displays correctly in the plot after reshaping, as unusual dimensions or large file sizes may require additional adjustments.
