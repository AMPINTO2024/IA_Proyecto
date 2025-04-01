from fer import FER
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd

scripts_path = os.getcwd()
project_path = os.path.dirname(scripts_path)

def evaluate_folder(folder_path):
    photos = os.listdir(folder_path)
    df = pd.DataFrame()
    for photo in photos:
        photo_path = os.path.join(folder_path, photo)
        img = cv2.imread(photo_path)
        if img is None:
            continue

        # Convert the image from BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Initialize the FER detector
        detector = FER()

        # Detect emotions in the image
        result = detector.detect_emotions(img_rgb)
        if len(result) == 0:
            continue

        # Get the emotions
        emotions = result[0]['emotions']

        # Append the emotions to the DataFrame
        df_emotions = pd.DataFrame(emotions, index=[photo])
        df = pd.concat([df, df_emotions], ignore_index=True)

    return df


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        print("No existe la carpeta")
        return False
    df = evaluate_folder(folder_path)
    # average emotions
    df_mean = df.mean() * 100
    # plot the results as a bar chart, rotate horizontal bars
    df_mean.plot(kind='barh')
    plt.show()


def main(folder_name):
    dataset_path = os.path.join(project_path, 'dataset', 'emociones2')
    photos_path = os.path.join(dataset_path, 'train', folder_name)
    check_folder(photos_path)



if __name__ == '__main__':
    main('angry')