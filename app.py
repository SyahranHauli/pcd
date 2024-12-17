import os
import numpy as np
import cv2
import requests
import tarfile
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Image Processing Libraries
from skimage.feature import hog, local_binary_pattern
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Data Download and Extraction Functions
def download_dataset(url, filename):
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

def extract_dataset(filename, extract_path):
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall(path=extract_path)

# Feature Extraction Functions
def extract_hog_features(images):
    hog_features = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        fd = hog(gray, orientations=9,
                pixels_per_cell=(16, 16),
                cells_per_block=(2, 2),
                visualize=False)
        hog_features.append(fd)
    return np.array(hog_features)

def extract_sift_features(images, num_keypoints=100):
    sift = cv2.SIFT_create()
    sift_features = []

    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        if descriptors is not None:
            if len(keypoints) > num_keypoints:
                idx = np.random.choice(len(keypoints), num_keypoints, replace=False)
                descriptors = descriptors[idx]
            if len(keypoints) < num_keypoints:
                padding = np.zeros((num_keypoints - len(keypoints), 128))
                descriptors = np.vstack([descriptors, padding]) if descriptors is not None else padding
        else:
            descriptors = np.zeros((num_keypoints, 128))

        sift_features.append(descriptors.flatten())

    return np.array(sift_features)

def extract_lbp_features(images):
    lbp_features = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(),
                             bins=np.arange(0, n_points + 3),
                             range=(0, n_points + 2))
        hist = hist.astype('float') / hist.sum()
        lbp_features.append(hist)
    return np.array(lbp_features)

# Feature Scaling
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def load_dataset(path, num_images=1000):
    images = []
    labels = []
    classes = []
    count = 0

    for image_file in os.listdir(path):
        if count >= num_images:
            break

        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(path, image_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            images.append(img)
            label = ' '.join(image_file.split('_')[:-1])
            labels.append(label)

            if label not in classes:
                classes.append(label)
            count += 1

    return np.array(images), np.array(labels), classes

# Feature Visualization Functions
def plot_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    fd, hog_image = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                        cells_per_block=(2, 2), visualize=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    ax2.imshow(hog_image, cmap='gray')
    ax2.set_title('HOG Features')
    ax2.axis('off')
    return fig

def plot_sift_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    sift_image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(sift_image)
    plt.title('SIFT Keypoints')
    plt.axis('off')
    return fig

def plot_lbp_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    ax2.imshow(lbp, cmap='gray')
    ax2.set_title('LBP Features')
    ax2.axis('off')
    return fig

def main():
    st.title('Pet Image Classification')
    st.markdown('Explore image feature extraction and classification techniques')

    # Check if dataset exists
    dataset_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    dataset_filename = "images.tar.gz"
    extract_path = "oxford-iiit-pet"
    dataset_path = os.path.join(extract_path, "images")

    if not os.path.exists(extract_path):
        st.info("Downloading dataset...")
        download_dataset(dataset_url, dataset_filename)
        st.info("Extracting dataset...")
        extract_dataset(dataset_filename, extract_path)
        os.remove(dataset_filename)

    # Load Dataset
    X, y, classes = load_dataset(dataset_path)
    
    # Encoding Labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Feature Extraction Options
    feature_option = st.sidebar.selectbox(
        'Select Feature Extraction Method',
        ['HOG', 'SIFT', 'LBP']
    )

    # Extraction and Scaling Based on Selection
    if feature_option == 'HOG':
        X_train_features = extract_hog_features(X_train)
        X_test_features = extract_hog_features(X_test)
    elif feature_option == 'SIFT':
        X_train_features = extract_sift_features(X_train)
        X_test_features = extract_sift_features(X_test)
    else:  # LBP
        X_train_features = extract_lbp_features(X_train)
        X_test_features = extract_lbp_features(X_test)

    # Scale Features
    X_train_scaled, X_test_scaled, _ = scale_features(X_train_features, X_test_features)

    # Model Training
    classifier_option = st.sidebar.selectbox(
        'Select Classifier',
        ['SVM', 'CNN']
    )

    if classifier_option == 'SVM':
        svm_classifier = SVC(kernel='rbf', random_state=42)
        svm_classifier.fit(X_train_scaled, y_train)
        y_pred = svm_classifier.predict(X_test_scaled)
    else:
        # Simple CNN
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(len(classes), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, validation_split=0.2, verbose=0)
        y_pred = model.predict(X_test).argmax(axis=1)

    # Performance Metrics
    st.header('Model Performance')
    st.text(f'Accuracy: {accuracy_score(y_test, y_pred):.2%}')
    st.text(classification_report(y_test, y_pred, target_names=[classes[i] for i in range(len(classes))]))

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[classes[i] for i in range(len(classes))],
                yticklabels=[classes[i] for i in range(len(classes))])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)

    # Feature Visualization
    st.header('Feature Extraction Visualization')
    sample_image = X_train[0]
    if feature_option == 'HOG':
        st.pyplot(plot_hog_features(sample_image))
    elif feature_option == 'SIFT':
        st.pyplot(plot_sift_features(sample_image))
    else:
        st.pyplot(plot_lbp_features(sample_image))

if __name__ == "__main__":
    # Suppress TensorFlow warnings
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    main()
