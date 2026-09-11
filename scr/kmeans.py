import numpy as np
import matplotlib.pyplot as plt
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

# Choose number of colors
K = 8

# Simplified K-Means from scratch (just 4 steps!)
centroids = pixels[np.random.choice(len(pixels), K, replace=False)]  

# Step 1: Pick random leaders
for _ in range(10):  # Repeat 10 times
    # Step 2: Assign each pixel to nearest centroid
    distances = np.sqrt(((pixels - centroids[:, np.newaxis])**2).sum(axis=2))
    labels = np.argmin(distances, axis=0)
    
    # Step 3: Compute new centroids (average of each group)
    new_centroids = np.array([pixels[labels == i].mean(axis=0) for i in range(K)])
    
    # Step 4: Check if centroids changed
    if np.all(centroids == new_centroids):
        break
    centroids = new_centroids
    
# Get palette and reconstruct image
palette = centroids.astype(int)
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
    
    
'''
What the code does:
Your Code	What it does
centroids = pixels[np.random.choice(...)]          Picks K random pixels as initial "leaders"
distances = np.sqrt(((pixels - centroids[:, np.newaxis])**2).sum(axis=2))	Calculates distance from every pixel to every centroid
labels = np.argmin(distances, axis=0)	Assigns each pixel to its nearest centroid
new_centroids = np.array([pixels[labels == i].mean(axis=0) for i in range(K)])	Averages all pixels in each group to find new centroids
if np.all(centroids == new_centroids): break	Stops when centroids stop moving
'''