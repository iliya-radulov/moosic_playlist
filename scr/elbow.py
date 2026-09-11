import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2

# Load image
image_path = "me.jpg"
image = cv2.imread(image_path)
if image is None:
    print("ERROR: Image not found!")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pixels = image_rgb.reshape(-1, 3)

# Sample pixels to speed things up (use 10,000 random pixels)
if len(pixels) > 10000:
    np.random.seed(42)
    indices = np.random.choice(len(pixels), 10000, replace=False)
    sample_pixels = pixels[indices]
else:
    sample_pixels = pixels

# Test K from 2 to 20
k_values = range(2, 21)
inertias = []

print("Testing K values...")
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
    kmeans.fit(sample_pixels)
    inertias.append(kmeans.inertia_)  # inertia = how "tight" the clusters are
    print(f"K={k}: {kmeans.inertia_:.0f}")

# Plot the elbow curve
plt.figure(figsize=(10, 5))
plt.plot(k_values, inertias, 'bo-')
plt.xlabel('Number of Colors (K)')
plt.ylabel('Inertia (Lower is better)')
plt.title('Elbow Method - Find Optimal K')
plt.grid(True)
plt.show()

# Ask user for K
best_k = int(input("Based on the elbow graph, enter your chosen K: "))

# Run with chosen K
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans.fit(pixels)

palette = kmeans.cluster_centers_.astype(int)
labels = kmeans.labels_
compressed_pixels = palette[labels]
compressed_image = compressed_pixels.reshape(image_rgb.shape)

# Show results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image_rgb)
axes[0].set_title("Original")
axes[0].axis('off')

axes[1].imshow(compressed_image.astype(np.uint8))
axes[1].set_title(f"Compressed ({best_k} colors)")
axes[1].axis('off')

palette_display = np.array([palette] * 100)
axes[2].imshow(palette_display.astype(np.uint8))
axes[2].set_title("Color Palette")
axes[2].axis('off')

plt.tight_layout()
plt.show()

print(f"\nPalette colors:")
for i, color in enumerate(palette):
    hex_code = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
    print(f"  {hex_code}")