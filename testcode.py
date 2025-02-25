#data_it is a tensor in the shape, [128, C, H, W], so it should be [128,1,28,28]

test_img = data_it[0] # should pull the first image 

test_img = test_img.squeeze(0) #gets rid of the channel dimension

print(test_img.size())
plt.imshow(test_img.numpy(), cmap="gray")
plt.imshow(test_img.numpy(), cmap="gray")  # Convert to NumPy and plot
plt.title(f"Label: {targets_it[0].item()}")  # Display the corresponding label
plt.axis("off")  # Hide axes
plt.show()
