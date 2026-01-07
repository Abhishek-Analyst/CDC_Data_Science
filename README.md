# CDC_Data_Science
 Satellite Imagery Based Property Valuation (Multimodal Regression)
 
- Project Overview
-This project aims to predict residential property prices using a multimodal machine learning approach by combining:
-Tabular property data (size, rooms, location-related features)
-Satellite imagery capturing neighborhood and environmental context
-Unlike traditional models that rely only on structured data, this project integrates visual information from satellite
images to enhance understanding of property surroundings such as road connectivity, building density, and green cover.

The project is divided into two major stages.

1. Tabular Baseline Model
In the first stage, a regression model is trained using only structured numerical data such as:
Number of bedrooms and bathrooms
Living area and plot size
Condition, grade, and view
Geographic coordinates (latitude & longitude)
This model acts as a baseline to compare performance later.
2. Multimodal Model (Tabular + Satellite Images)
In the second stage:
Satellite images are collected using latitude and longitude values
A pretrained ResNet CNN is used to extract features from these images
Image features are merged with tabular features
A regression model is trained on this combined feature set
Grad-CAM is applied to understand which regions of images influence the model
This helps in adding visual neighborhood context to price prediction.


Repository Structure
├── cdc.ipynb              # Complete end-to-end project notebook
├── images/
│   └── train_subset/      # Satellite images used for multimodal modeling
├── submission.csv         # Final predictions on test data
└── README.md              # Project documentation


Notebook Description (cdc.ipynb)

The notebook contains the entire workflow, organized step by step:

Data loading and basic inspection

Exploratory Data Analysis (EDA)

Feature engineering

Training the tabular baseline model

Satellite image generation

Image preprocessing

CNN-based feature extraction

Feature fusion (tabular + image features)

Multimodal model training and evaluation

Grad-CAM based explainability

Final prediction generation

Each section is clearly commented and explained in simple language.

Dataset Details
Tabular Data

Includes features like:

Bedrooms, bathrooms

Living area, lot size

Floors, condition, grade

Waterfront and view

Latitude and longitude

Neighborhood-related features

Additional engineered features:

Sale year and month

House age

Renovation flag

Land utilization ratio

Visual Data

Satellite images are fetched using property coordinates.
These images capture:

Neighborhood structure

Green areas

Surrounding infrastructure


| Model Type                    | RMSE                | R² Score |
| ----------------------------- | ------------------  | -------- |
| Tabular Baseline              | 126461.20019878604  | 0.87    |
| Multimodal (Tabular + Images) | 174919.0024793155   | 0.758 |

The tabular-only model performs better numerically.
The multimodal model adds visual explainability and contextual understanding.

Explainability (Grad-CAM)

Grad-CAM is used to highlight important regions in satellite images.
The attention maps mainly focus on:

Residential zones

Green spaces

Overall neighborhood layout

This confirms that the CNN learns meaningful environmental patterns.


Author

This project was developed as part of a data science and multimodal machine learning assignment,
demonstrating how computer vision and tabular data can be combined for real-world house price prediction.

