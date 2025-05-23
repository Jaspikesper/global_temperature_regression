# Interactive Regression Explorer - Climate Visualization Suite

Welcome to the Math Club's Climate Change Project. This toolkit lets you visualize climate trends interactively, offering both a GUI application and an animated multi-fit explorer. Experiment with different regression models, explore datasets, and even overlay a custom background image (like the sun).

---

## Launching the GUI

Run the following to launch the interactive regression GUI:

```bash
python subplot.py
```

This opens a window where you can:

- Choose between datasets: Temperature, CO2 & Temperature, and GIS records
- Select fitting methods: Linear, Polynomial (up to Quartic), Exponential, or LOESS smoothing
- Control how far into the future the regression extrapolates
- Adjust scatter-point size
- Toggle a background image, ideally shaped like a sun

Mouse over any point to see real-time tooltips showing observed vs. predicted values. Drag vertically to see effects live if dragging is supported in your build.

Note: If the GUI runs slowly or becomes unresponsive, tell Jasper to optimize the redraw loop or lower the scatter-point count. He's probably already got a fix.

---

## Running the Grid Explorer

Try the interactive regression grid in main.py, which compares multiple models side-by-side:

```bash
python main.py
```

This tool:

- Displays multiple regression fits (Linear, Exponential, LOESS, etc.)
- Responds to mouse hover with precise model diagnostics
- Supports real-time prediction past observed years (dashed line)
- Makes it easier to compare model quality at a glance

---

## Image Preparation

To include a custom background (like the sun), format your image with aspect_reshaper.py.

### Step-by-Step

1. Place your original image in the static/ directory.  
2. Edit the script's top lines to reference your filename:

   ```python
   input_file = 'static/your_image.png'
   output_file = 'static/useMe.png'
   target_aspect_ratio = 9 / 16
   ```

3. Run the script:

   ```bash
   python aspect_reshaper.py
   ```

4. Your image will be stretched and resized. It will automatically show up as a background if enabled in the GUI.

---

## Data Overview

data_loader.py handles three datasets:

- Temperature_Data.csv - historical climate anomalies  
- merged_co2_temp.csv - CO2 and temperature combined data  
- gistemp.csv - long-term NASA GIS records  

These live in the data/ folder and are normalized to lowercase for column safety.

---

## Notes

- All visualizations use matplotlib with the TkAgg backend for interactivity.  
- Install dependencies with:

  ```bash
  pip install numpy scipy statsmodels matplotlib pandas
  ```

- macOS users: you may need to invoke pythonw to run the GUI properly.  
- For rubric and planning documents, see the project_high-levels/ folder.

---

Enjoy exploring climate trends - and don't forget to bug Jasper if things get janky!

