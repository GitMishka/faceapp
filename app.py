import cv2
import tkinter as tk
from tkinter import filedialog


def load_image(file_path):
    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


def detect_faces(gray_image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    return faces


def draw_faces(image, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return image


from PIL import Image, ImageTk

def display_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)

    window = tk.Toplevel()
    window.title("Face Detection")
    tk_image = ImageTk.PhotoImage(pil_image)
    label = tk.Label(window, image=tk_image)
    label.pack()
    window.mainloop()

    
def main():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])

    if file_path:
        gray_image = load_image(file_path)
        faces = detect_faces(gray_image)
        image_with_faces = draw_faces(cv2.imread(file_path), faces)
        display_image(image_with_faces)
        
if __name__ == "__main__":
    main()


