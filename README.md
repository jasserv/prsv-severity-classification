# PRSV Severity Classification with Raspberry Pi Deployment

This project presents a deep learning-based solution for Papaya Ringspot Virus (PRSV) severity classification, integrated into a real-time hardware system using Raspberry Pi 5. The system leverages CNN models to classify papaya leaf images based on severity levels, ultimately aiding in efficient field diagnosis and disease monitoring.

📌 Highlights
Dataset: 3,000 augmented and balanced RGB images (256x256 resolution), labeled based on a 5-level severity scale as defined by Alviar et al., 2012.

Models Tested: Various CNN architectures including InceptionV3 and EfficientNet variants.

Top Performer: EfficientNetLite outperformed all tested models with:

Recall: 88.75%

Precision: 89.44%

Accuracy: 88.71%

F1-score: 88.85%

Deployment: Optimized and deployed on Raspberry Pi 5 for real-time inference with responsive UI display and field-ready performance.
