import glob
import numpy as np
import cv2 as cv
import csv
import os
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC

def adjust_brightness(image, min_brightness=0.4, max_brightness=0.6):
    x, y = image.shape[:2]
    current_brightness = np.sum(image) / (255 * x * y)

    while(current_brightness < min_brightness or current_brightness > max_brightness):
        brightness_adjustment = round(((255 * x * y * 0.5) - np.sum(image)) / (x * y))
        adjustment_array = np.ones((x, y), dtype=int) * brightness_adjustment
        image = np.clip(np.add(image, adjustment_array), 0, 255)
        current_brightness = np.sum(image) / (255 * x * y)
    
    return image

def preprocess_images(image_set, input_dir, output_dir):
    image_names = [path.rsplit('\\', 1)[-1] for path in glob.glob(os.path.join(input_dir, image_set, '*.jpg'))]
    images = [cv.imread(file, cv.IMREAD_GRAYSCALE) for file in glob.glob(os.path.join(input_dir, image_set, '*.jpg'))]

    for i, image in enumerate(images):
        images[i] = adjust_brightness(image)

    image_sizes = [50, 200]
    for size in image_sizes:
        for i, image in enumerate(images):
            resized_image = cv.resize(image, (size, size), interpolation=cv.INTER_LINEAR_EXACT).astype(np.uint8)
            output_path = os.path.join(output_dir, image_set, str(size), image_names[i])
            cv.imwrite(output_path, resized_image)

image_sets = ['bedroom', 'coast', 'forest']
input_dir = './ProjData'
output_dir = './AnalysisData'

for image_set in image_sets:
    preprocess_images(image_set, input_dir+'/Train', output_dir+'/Train')
    preprocess_images(image_set, input_dir+'/Test', output_dir+'/Test')

# Extract SIFT features on ALL training images and save the data.
def extract_sift_features(image_set, resize, images, is_train=True):
    sift = cv.SIFT_create()

    sift_features = []
    max_length = 0

    for image in images:
        keypoints, descriptors = sift.detectAndCompute(image, None)
        if descriptors is not None:
            length = descriptors.shape[0]
            if length > max_length:
                max_length = length
            sift_features.append(descriptors)
    
    # Pad the shorter arrays with zeros
    for i in range(len(sift_features)):
        descriptors = sift_features[i]
        length = descriptors.shape[0]
        if length < max_length:
            padded_descriptors = np.zeros((max_length, descriptors.shape[1]))
            padded_descriptors[:length, :] = descriptors
            sift_features[i] = padded_descriptors
    
    
    sift_features = np.array(sift_features)
    sift_features = sift_features.astype(np.float32)

    if is_train:
        # Write the features to a CSV file for training images
        with open(f'./AnalysisData/Train/{image_set}/sift{resize}_features.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(sift_features)
    else:
        # Write the features to a CSV file for testing images
        with open(f'./AnalysisData/Test/{image_set}/sift{resize}_features.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(sift_features)


# Extract Histogram features on ALL training images and save the data
def extract_histogram_features(images, image_set, size, is_train=True):
    features = []
    for image in images:
        resized = cv.resize(image, size, interpolation=cv.INTER_LINEAR_EXACT)
        hist = cv.calcHist([resized], [0], None, [256], [0, 256])
        features.append(hist.flatten())
    if is_train:
        with open(f'./AnalysisData/Train/{image_set}/histfeatures_{size[0]}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(features)
    else:
        with open(f'./AnalysisData/Test/{image_set}/histfeatures_{size[0]}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(features)

for image_set in image_sets:
    train_images50 = [cv.imread(file) for file in glob.glob('./AnalysisData/Train/' + image_set + "/50/" + '/*.jpg')]
    train_images200 = [cv.imread(file) for file in glob.glob('./AnalysisData/Train/' + image_set + "/200/" + '/*.jpg')]
    test_images50 = [cv.imread(file) for file in glob.glob('./AnalysisData/Test/' + image_set + "/50/" + '/*.jpg')]
    test_images200 = [cv.imread(file) for file in glob.glob('./AnalysisData/Test/' + image_set + "/200/" + '/*.jpg')]
    extract_sift_features(image_set, 200, train_images200)
    extract_sift_features(image_set, 50, train_images50)
    extract_sift_features(image_set, 200, test_images200, is_train=False)
    extract_sift_features(image_set, 50, test_images50, is_train=False)

    extract_histogram_features(train_images200, image_set, (200, 200))
    extract_histogram_features(train_images50, image_set, (50, 50))
    extract_histogram_features(test_images200, image_set, (200, 200), is_train=False)
    extract_histogram_features(test_images50, image_set, (50, 50), is_train=False)

# Define the labels for each image set
label_dict = {'bedroom': 0, 'coast': 1, 'forest': 2}

#4a
# Read the training images
train_images = []
train_labels = []
for image_set in image_sets:
    for image_path in glob.glob(f"./AnalysisData/Train/{image_set}/50/*.jpg"):
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        train_images.append(image.flatten())
        train_labels.append(label_dict[image_set])

# Read the test images
test_images = []
test_labels = []
for image_set in image_sets:
    for image_path in glob.glob(f"./AnalysisData/Test/{image_set}/50/*.jpg"):
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        image = adjust_brightness(image)
        image = cv.resize(image, (50, 50), interpolation=cv.INTER_LINEAR_EXACT)
        test_images.append(image.flatten())
        test_labels.append(label_dict[image_set])

# Train the Nearest Neighbor classifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train_images, train_labels)

# Predict the labels of the test images
predicted_labels = knn.predict(test_images)

# Calculate the confusion matrix
cm = confusion_matrix(test_labels, predicted_labels)

# Calculate the percentage of false negatives
false_negatives_pct = 100 * (len(test_images) - cm.sum())/len(test_images)

# Calculate the percentage of correctly classified images
correct_pct = 100 * (cm.diagonal().sum() / len(test_images))

# Calculate the percentage of false positives
false_positives_pct = 100 - correct_pct - false_negatives_pct

print("\n50*50 pixel values with Nearest Neighbor classifier:")
print(f"Percentage of correctly classified images: {correct_pct:.2f}%")
print(f"Percentage of false positives: {false_positives_pct:.2f}%")
print(f"Percentage of false negatives: {false_negatives_pct:.2f}%")

#4b
# Load the SIFT features of the training images
train_data = []
train_labels = []
for image_set in image_sets:
    for size in [50, 200]:
        with open(f'./AnalysisData/Train/{image_set}/sift{size}_features.csv') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                values = [float(x) for x in row[0][1:-1].split()]
                train_data.append(np.array(values))
                train_labels.append(label_dict[image_set])

train_data = np.array(train_data)
train_labels = np.array(train_labels)

# Load the SIFT features of the test images
test_images = []
test_labels = []
for image_set in image_sets:
    for size in [50, 200]:
        with open(f'./AnalysisData/Test/{image_set}/sift{size}_features.csv') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                values = [float(x) for x in row[0][1:-1].split()]
                test_images.append(np.array(values))
                test_labels.append(label_dict[image_set])

test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Fit the Nearest Neighbor model to the training data
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train_data, train_labels)

# Predict the labels of the test data
predicted_labels = knn.predict(test_images)

# Calculate the confusion matrix
cm = confusion_matrix(test_labels, predicted_labels)
# Calculate the percentage of false negatives
false_negatives_pct = 100 * (len(test_images) - cm.sum())/len(test_images)

# Calculate the percentage of correctly classified images
correct_pct = 100 * (cm.diagonal().sum() / len(test_images))

# Calculate the percentage of false positives
false_positives_pct = 100 - correct_pct - false_negatives_pct

print("\nSIFT feature data with Nearest Neighbor classifier:")
print(f"Percentage of correctly classified images: {correct_pct:.2f}%")
print(f"Percentage of false positives: {false_positives_pct:.2f}%")
print(f"Percentage of false negatives: {false_negatives_pct:.2f}%")

#4c
# Load the histogram features of the training images
train_data = []
train_labels = []
for image_set in image_sets:
    for size in [50, 200]:
        with open(f'./AnalysisData/Train/{image_set}/histfeatures_{size}.csv') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                train_data.append(np.array(row).astype(float))
                train_labels.append(label_dict[image_set])

train_data = np.array(train_data)
train_labels = np.array(train_labels)

# Load the histogram features of the test images
test_images = []
test_labels = []
for image_set in image_sets:
    for size in [50, 200]:
        with open(f'./AnalysisData/Test/{image_set}/histfeatures_{size}.csv') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                test_images.append(np.array(row).astype(float))
                test_labels.append(label_dict[image_set])

test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Fit the Nearest Neighbor model to the training data
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train_data, train_labels)

# Predict the labels of the test data
predicted_labels = knn.predict(test_images)

# Calculate the confusion matrix
cm = confusion_matrix(test_labels, predicted_labels)

# Calculate the percentage of false negatives
false_negatives_pct = 100 * (len(test_images) - cm.sum())/len(test_images)

# Calculate the percentage of correctly classified images
correct_pct = 100 * (cm.diagonal().sum() / len(test_images))

# Calculate the percentage of false positives
false_positives_pct = 100 - correct_pct - false_negatives_pct

print("\nHistogram feature data with Nearest Neighbor classifier:")
print(f"Percentage of correctly classified images: {correct_pct:.2f}%")
print(f"Percentage of false positives: {false_positives_pct:.2f}%")
print(f"Percentage of false negatives: {false_negatives_pct:.2f}%")

#4d
# Load the SIFT features of the training images
train_data = []
train_labels = []
for image_set in image_sets:
    for size in [50, 200]:
        with open(f'./AnalysisData/Train/{image_set}/sift{size}_features.csv') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                values = [float(x) for x in row[0][1:-1].split()]
                train_data.append(np.array(values))
                train_labels.append(label_dict[image_set])

train_data = np.array(train_data)
train_labels = np.array(train_labels)

# Load the SIFT features of the test images
test_images = []
test_labels = []
for image_set in image_sets:
    for size in [50, 200]:
        with open(f'./AnalysisData/Test/{image_set}/sift{size}_features.csv') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                values = [float(x) for x in row[0][1:-1].split()]
                test_images.append(np.array(values))
                test_labels.append(label_dict[image_set])

test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Fit the Linear SVM model to the training data
svm = LinearSVC()
svm.fit(train_data, train_labels)

# Predict the labels of the test data
predicted_labels = svm.predict(test_images)

# Calculate the confusion matrix
cm = confusion_matrix(test_labels, predicted_labels)
# Calculate the percentage of false negatives
false_negatives_pct = 100 * (len(test_images) - cm.sum())/len(test_images)

# Calculate the percentage of correctly classified images
correct_pct = 100 * (cm.diagonal().sum() / len(test_images))

# Calculate the percentage of false positives
false_positives_pct = 100 - correct_pct - false_negatives_pct

print("\nSIFT feature data with linear SVM classifier:")
print(f"Percentage of correctly classified images: {correct_pct:.2f}%")
print(f"Percentage of false positives: {false_positives_pct:.2f}%")
print(f"Percentage of false negatives: {false_negatives_pct:.2f}%")
