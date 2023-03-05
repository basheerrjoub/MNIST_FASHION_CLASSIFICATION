from utils import mnist_reader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import random
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import ndimage
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.manifold import TSNE
import warnings
from random import shuffle
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report
import numpy as np
from skimage.feature import hog
import winsound

warnings.filterwarnings("ignore", category=FutureWarning)
root = tk.Tk()
root.title("ML Project")

# Dict of clothes [TYPES]
types = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}

X_train, y_train = mnist_reader.load_mnist("data/fashion", kind="train")
X_test, y_test = mnist_reader.load_mnist("data/fashion", kind="t10k")

# X_train, y_train = X_train[:400], y_train[:400]
# X_test, y_test = X_test[:200], y_test[:200]
# X_train, X_test, y_train, y_test = train_test_split(
#     X_train,
#     y_train,
#     test_size=0.34,
#     random_state=1,
#     shuffle=True,
# )
# X_test, y_test = X_train[400:600], y_train[400:600]
# X_train, y_train = X_train[:400], y_train[:400]
################################################################################################################################################################################
example_hog = []


def calc_hog_features(X, image_shape=(28, 28), pixels_per_cell=(4, 4)):
    fd_list = []
    for row in X:
        img = row.reshape(*image_shape)
        fd = hog(
            img,
            orientations=9,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=(2, 2),
        )
        fd_list.append(fd)

    return np.array(fd_list)


X_train = calc_hog_features(X_train)
X_test = calc_hog_features(X_test)
print("#Features: ", len(X_train[0]))

# Plot the HOG features as a histogram
fig, ax = plt.subplots()
ran = random.randint(1, len(X_train))
img = X_train[ran][:]
img = plt.hist(img)
plt.title(f"HOG for {types[y_train[ran]]}")
plt.close()
################################################################################################################
# Combine X and y into a single array
train_data = np.column_stack((X_train, y_train))
test_data = np.column_stack((X_test, y_test))

# Shuffle the arrays
np.random.seed(42)
np.random.shuffle(train_data)
np.random.seed(42)
np.random.shuffle(test_data)

# Split the shuffled arrays back into X and y
X_train = train_data[:, :-1]
y_train = train_data[:, -1]
X_test = test_data[:, :-1]
y_test = test_data[:, -1]


X_test_v = X_test
winsound.Beep(2500, 2000)
#  __      ___                 _ _
#  \ \    / (_)               | (_)
#   \ \  / / _ ___ _   _  __ _| |_ _______
#    \ \/ / | / __| | | |/ _` | | |_  / _ \
#     \  /  | \__ \ |_| | (_| | | |/ /  __/
#      \/   |_|___/\__,_|\__,_|_|_/___\___|


def visualize_function(X_test):
    # Reshape and normalize data
    X_test = X_test.reshape(X_test.shape[0], 72)
    X_test = X_test
    # Apply t-SNE to the data
    tsne = TSNE(n_components=2, perplexity=10, random_state=0)
    x_tsne = tsne.fit_transform(X_test)

    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y_test)
    plt.colorbar()
    plt.show()


#   ____           _____ ______   _      _____ _   _ ______   _  ___   _ _   _
#  |  _ \   /\    / ____|  ____| | |    |_   _| \ | |  ____| | |/ / \ | | \ | |
#  | |_) | /  \  | (___ | |__    | |      | | |  \| | |__    | ' /|  \| |  \| |
#  |  _ < / /\ \  \___ \|  __|   | |      | | | . ` |  __|   |  < | . ` | . ` |
#  | |_) / ____ \ ____) | |____  | |____ _| |_| |\  | |____  | . \| |\  | |\  |
#  |____/_/    \_\_____/|______| |______|_____|_| \_|______| |_|\_\_| \_|_| \_|
#
def KNN_function():
    knn_model = KNeighborsClassifier(n_neighbors=3, metric="manhattan")
    knn_model.fit(X_train, y_train)
    predict = knn_model.predict(X_test)

    # Compute Accuracy
    true_predicted = 0
    for i in range(len(predict)):
        if y_test[i] == predict[i]:
            true_predicted += 1
    accuracy_KNN = true_predicted / len(predict)
    accuracy_KNN = round(accuracy_KNN, 5)
    print("Accuracy[BASE MODEL]: ", accuracy_KNN)
    con_matrix = pd.crosstab(
        pd.Series(y_test.ravel(), name="Actual"), pd.Series(predict, name="Predicted")
    )
    fig = plt.figure(figsize=(8, 5))
    plt.title("Confusion Matrix for KNN")
    sns.heatmap(con_matrix, cmap="Blues", annot=True, fmt="g")
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=5, column=0, padx=10)
    winsound.Beep(2500, 2000)
    return accuracy_KNN


#   _____                 _                   ______                  _
#  |  __ \               | |                 |  ____|                | |
#  | |__) |__ _ _ __   __| | ___  _ __ ___   | |__ ___  _ __ ___  ___| |_
#  |  _  // _` | '_ \ / _` |/ _ \| '_ ` _ \  |  __/ _ \| '__/ _ \/ __| __|
#  | | \ \ (_| | | | | (_| | (_) | | | | | | | | | (_) | | |  __/\__ \ |_
#  |_|  \_\__,_|_| |_|\__,_|\___/|_| |_| |_| |_|  \___/|_|  \___||___/\__|
def RF_function():

    random_forest = RandomForestClassifier(
        criterion="entropy",
        max_depth=122,
        n_estimators=59,
        max_features="sqrt",
    )
    random_forest.fit(X_train, y_train)

    predict = random_forest.predict(X_test)
    true_predicted = 0
    for i in range(len(predict)):
        if y_test[i] == predict[i]:
            true_predicted += 1
    accuracy_RF = true_predicted / len(predict)
    accuracy_RF = round(accuracy_RF, 5)
    print("Accuracy[RF]: ", accuracy_RF)
    # Compute the Accuracy for each class:
    acc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    class_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    wrong = []
    for i in range(len(predict)):
        if y_test[i] == 0:
            class_count[0] += 1
            if y_test[i] == predict[i]:
                acc[0] += 1
        elif y_test[i] == 1:
            class_count[1] += 1
            if y_test[i] == predict[i]:
                acc[1] += 1
        elif y_test[i] == 2:
            class_count[2] += 1
            if y_test[i] == predict[i]:
                acc[2] += 1
        elif y_test[i] == 3:
            class_count[3] += 1
            if y_test[i] == predict[i]:
                acc[3] += 1
        elif y_test[i] == 4:
            class_count[4] += 1
            if y_test[i] == predict[i]:
                acc[4] += 1
        elif y_test[i] == 5:
            class_count[5] += 1
            if y_test[i] == predict[i]:
                acc[5] += 1
        elif y_test[i] == 6:
            class_count[6] += 1
            if y_test[i] == predict[i]:
                acc[6] += 1
            else:  # Here is the analysis for the wrong predictions of class 6
                wrong.append(f"Index: {i}, Predicted: {types[predict[i]]}")

        elif y_test[i] == 7:
            class_count[7] += 1
            if y_test[i] == predict[i]:
                acc[7] += 1
        elif y_test[i] == 8:
            class_count[8] += 1
            if y_test[i] == predict[i]:
                acc[8] += 1
        elif y_test[i] == 9:
            class_count[9] += 1
            if y_test[i] == predict[i]:
                acc[9] += 1
    print(
        f"ACCURACY: C0: {acc[0] / class_count[0]}, C1: {acc[1] / class_count[1]}, C2: {acc[2] / class_count[2]}, C3: {acc[3] / class_count[3]}, C4: {acc[4] / class_count[4]}, C5: {acc[5] / class_count[5]}, C6: {acc[6] / class_count[6]}, C7: {acc[7] / class_count[7]}, C8: {acc[8] / class_count[8]}, C9: {acc[9] / class_count[9]}"
    )
    with open("wrong.txt", "w") as f:
        for lst in wrong:
            f.write(str(lst) + "\n")
    con_matrix = pd.crosstab(
        pd.Series(y_test.ravel(), name="Actual"), pd.Series(predict, name="Predicted")
    )
    print(classification_report(y_test, predict))

    fig = plt.figure(figsize=(8, 5))
    plt.title("Confusion Matrix for RF")
    sns.heatmap(con_matrix, cmap="Blues", annot=True, fmt="g")
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=5, column=0, padx=10)
    plt.close()
    winsound.Beep(2500, 2000)
    return accuracy_RF


#    _______      ____  __
#   / ____\ \    / /  \/  |
#  | (___  \ \  / /| \  / |
#   \___ \  \ \/ / | |\/| |
#   ____) |  \  /  | |  | |
#  |_____/    \/   |_|  |_|


def SVM_function():

    SVM_model = SVC(C=13, kernel="rbf", gamma="auto", probability=True)
    SVM_model.fit(X_train, y_train)
    predict = SVM_model.predict(X_test)

    # Compute Accuracy
    true_predicted = 0
    for i in range(len(predict)):
        if y_test[i] == predict[i]:
            true_predicted += 1
    accuracy_SVM = true_predicted / len(predict)
    accuracy_SVM = round(accuracy_SVM, 2)
    print("Accuracy[SVM]: ", accuracy_SVM)
    con_matrix = pd.crosstab(
        pd.Series(y_test.ravel(), name="Actual"), pd.Series(predict, name="Predicted")
    )
    fig = plt.figure(figsize=(8, 5))
    plt.title("Confusion Matrix for SVM")
    sns.heatmap(con_matrix, cmap="Blues", annot=True, fmt="g")
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=5, column=0, padx=10)
    return accuracy_SVM


#    _____ _    _ _____
#   / ____| |  | |_   _|
#  | |  __| |  | | | |
#  | | |_ | |  | | | |
#  | |__| | |__| |_| |_
#   \_____|\____/|_____|


def show_accuracy_SVM():
    label.config(text=f"SVM ACC: {SVM_function()}", font=("Arial", 13), fg="green")


def show_accuracy_KNN():
    label.config(text=f"KNN ACC: {KNN_function()}", font=("Arial", 13), fg="red")


def show_accuracy_RF():
    label.config(text=f"RF ACC: {RF_function()}", font=("Arial", 13), fg="blue")


button1 = tk.Button(root, text="SVM ACCURACY", command=show_accuracy_SVM)
button1.grid(row=0, column=0, padx=5)
button2 = tk.Button(root, text="KNN ACCURACY", command=show_accuracy_KNN)
button2.grid(row=1, column=0, padx=5)
button3 = tk.Button(root, text="RF ACCURACY", command=show_accuracy_RF)
button3.grid(row=2, column=0, padx=5)
button4 = tk.Button(
    root, text="Visualize", command=lambda: visualize_function(X_test=X_test_v)
)
button4.grid(row=3, column=0, padx=0)


label = tk.Label(root)
label.grid(row=4, columnspan=2)

root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()


# Add the canvas to the Tkinter window
canvas.get_tk_widget().grid(row=5, column=0, padx=10)


x = (root.winfo_screenwidth() / 2) - (800 / 2)
y = (root.winfo_screenheight() / 2) - (650 / 2)
root.geometry("800x650+{}+{}".format(int(x), int(y)))

root.resizable(False, False)
root.mainloop()
