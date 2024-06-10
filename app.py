# Import beberapa library
import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image

# Fungsi untuk mendapatkan warna dominan sebanyak K
def get_dominant_colors(image, k=5):
    # Resize gambar untuk mempercepat proses
    image = image.resize((600, 400))

    # Mengkonversi gambar ke dalam bentuk array numpy
    image_array = np.array(image)

    # Reshape gambar ke dalam bentuk (jumlah piksel, 3)
    image = image_array.reshape((-1, 3))
    
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
    image = Image.open(uploaded_file)
    
    # Menampilkan gambar yang diupload ke website
    st.image(image, channels="RGB")

    # Mengambil warna dominan
    colors = get_dominant_colors(image)

    # Teks pada website
    st.write("Dominant Colors:")

    # Membuat dan menampilkan plot warna dominan
    fig = plot_colors(colors)
    st.pyplot(fig)
