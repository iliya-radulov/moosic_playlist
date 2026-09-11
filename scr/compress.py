import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2

# Load image
image_path = "me.jpg"  # Make sure this matches your filename
image = cv2.imread(image_path)

if image is None:
    print("ERROR: Could not load image. Make sure 'photo.jpg' is in the same folder.")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape pixels for K-Means
pixels = image_rgb.reshape(-1, 3)

# Choose number of colors
K = 8

# Run K-Means
print(f"Processing {len(pixels)} pixels with K={K}...")
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)

# See how "fast" K-Means runs
import time
start = time.time()
kmeans.fit(pixels)
print(f"Time taken: {time.time() - start:.2f} seconds")

kmeans.fit(pixels)

# Get palette and reconstruct image
palette = kmeans.cluster_centers_.astype(int)
labels = kmeans.labels_
compressed_pixels = palette[labels]
compressed_image = compressed_pixels.reshape(image_rgb.shape)

# Show results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original
axes[0].imshow(image_rgb)
axes[0].set_title("Original")
axes[0].axis('off')

# Compressed
axes[1].imshow(compressed_image.astype(np.uint8))
axes[1].set_title(f"Compressed ({K} colors)")
axes[1].axis('off')

# Palette
palette_display = np.array([palette] * 100)
axes[2].imshow(palette_display.astype(np.uint8))
axes[2].set_title("Color Palette")
axes[2].axis('off')

plt.tight_layout()
plt.show()

# Print hex codes
print("\nYour palette colors:")
for i, color in enumerate(palette):
    hex_code = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
    print(f"  {hex_code}")

# Save result
cv2.imwrite("me_compressed.jpg", cv2.cvtColor(compressed_image.astype(np.uint8), cv2.COLOR_RGB2BGR))
print(f"\nSaved compressed image as 'me_compressed.jpg'")