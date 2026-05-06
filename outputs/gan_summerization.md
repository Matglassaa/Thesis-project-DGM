---
name: gan-summarization
description: Act as an expert Machine Learning Engineer. Provide a comprehensive summarization of the exact training settings used to train the GAN model based on the `settings.json` file provided in the directory. After summarizing the initial settings, analyze the output realizations produced by the trained model by evaluating the loss metrics, facies distributions, and entropy figures. Output all results formatted as a markdown document.
---

**Task 1: Summarize Model Settings**
Extract and summarize all the initial model settings from `settings.json` by creating a well-formatted Markdown table. At the top of the output, generate a fitting title that highlights what makes these specific settings unique compared to other typical runs (e.g., *"Large dataset with high learning rate, 16 vertical grid cells"*).

**Task 2: Analyze Output Realizations**
After the table, write a concise technical analysis based on `loss_visualization.png`, `fluvgan_1_training_1_architecture_dcgan_4_1_history.csv`, `overall_facies_distribution.png`, and the three entropy PNG files. Address the following points explicitly:

1. **Loss Trends:** Analyze the Discriminator and Generator loss. Does one tend to go to zero while the other stays high (indicating potential mode collapse or overpowering)? Comment on the stability of the training and how often the losses explode or spike during the training sequence. Reference the CSV for exact values where helpful.
2. **Entropy Analysis:** Looking at the three entropy `.png` files, describe the spatial uncertainty. Is the entropy strictly higher than zero everywhere, or are there distinct regions of low/zero entropy? What does this imply about the model's confidence in those regions?

**Output Instructions:**
Provide the final output exclusively as valid Markdown content, structured cleanly with headers, so that it can be saved directly into a new file named `model_summarization.md`. Make sure to save this file in the same directory as the data contect files.