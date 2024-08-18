# Anomaly-Detection


# Video Anomaly Detection

This project aims to detect anomalies or criminal activities in video footage using deep learning techniques. Given a video input, the model can identify if there is any suspicious or criminal activity occurring, such as shoplifting, assault, robbery, or vandalism.

## Dataset

The project utilizes the [DCSASS Dataset](https://www.kaggle.com/datasets/mateohervas/dcsass-dataset), which contains a diverse collection of videos depicting various criminal and non-criminal activities. The dataset is divided into several categories, including "Normal" for non-criminal activities and different types of crimes like "Abuse," "Arson," "Assault," "Burglary," and more.

## Approach

The approach involves the following steps:

1. **Data Preprocessing**: The video frames are extracted and preprocessed using OpenCV. This includes resizing, normalization, and padding to ensure a consistent input format for the model.

2. **Feature Extraction**: A pre-trained ResNet-50 model is utilized for feature extraction from individual video frames.

3. **Model Architecture**: The extracted features are then fed into an LSTM (Long Short-Term Memory) network, which captures the temporal dynamics of the video sequences.

4. **Training**: The model is trained using the labeled videos from the dataset, with the goal of learning to classify video sequences into different categories (e.g., "Normal," "Shoplifting," "Assault," etc.).

5. **Prediction**: For a given input video, the model processes the video frames and outputs the predicted class, indicating whether the video contains any anomalous or criminal activity.

## Requirements

- Python 3.x
- PyTorch
- OpenCV
- scikit-learn
- numpy

## Usage

1. Clone the repository:

```
git clone https://github.com/your-username/video-anomaly-detection.git
```

2. Download the DCSASS Dataset from Kaggle and extract it into the project directory.


3. Run the Jupyter Notebook or Python script to train the model and make predictions on new videos.

## Evaluation

The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. These metrics are calculated by comparing the model's predictions with the ground truth labels from the dataset.

## Contributing

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The DCSASS Dataset creators and contributors
- The PyTorch, OpenCV, and scikit-learn communities

