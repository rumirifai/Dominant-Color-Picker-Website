# Import beberapa library
import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Fungsi untuk mendapatkan warna dominan sebanyak K
def get_dominant_colors(image, k=5):
    # Resize gambar untuk mempercepat proses
    image = cv2.resize(image, (600, 400))

    # Mengkonversi gambar dari BGR ke RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape gambar ke dalam bentuk (jumlah piksel, 3)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    
    # Menggunakan K-Means untuk menemukan warna dominan
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(image)

    # Mengambil warna puas (Centroids)
    colors = kmeans.cluster_centers_.astype(int)

    # Fungsi mereturn warna yang didapat
    return colors

# Fungsi untuk membuat visual warna dominan
def plot_colors(colors):
    # Membuat plot warna
    fig, ax = plt.subplots(1, figsize=(8, 2), subplot_kw=dict(xticks=[], yticks=[]))
    ax.imshow([colors])
    return fig

# Judul pada website
st.title("Dominant Color Picker")

# Fungsi untuk mengupload gambar ke website
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

# Kondisi ketika gambar telah di upload
if uploaded_file is not None:
    # Membaca file yang diunggah
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    # Mendekode gambar dengan OpenCV
    image = cv2.imdecode(file_bytes, 1)

    # Menampilkan gambar yang diupload ke website
    st.image(image, channels="BGR")

    # Mengambil warna dominan
    colors = get_dominant_colors(image)

    # Teks pada website
    st.write("Dominant Colors:")

    # Membuat dan menampilkan plow warna dominan
    fig = plot_colors(colors)
    st.pyplot(fig)
