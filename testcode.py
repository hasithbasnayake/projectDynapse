#data_it is a tensor in the shape, [128, C, H, W], so it should be [128,1,28,28]

test_img = data_it[0] # should pull the first image 

test_img = test_img.squeeze(0) #gets rid of the channel dimension

print(test_img.size())
plt.imshow(test_img.numpy(), cmap="gray")
plt.imshow(test_img.numpy(), cmap="gray")  # Convert to NumPy and plot
plt.title(f"Label: {targets_it[0].item()}")  # Display the corresponding label
plt.axis("off")  # Hide axes
plt.show()

# Maybe use channels to see if you can incorporate both the DoG off and DoG on images, so [128,2,28,28].
# You now know the size and that the images are being convolved correctly 

# data_it is a tensor

# spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01)

# # The function in SNNTorch convert_to_time(data, tau, threshold) allows us to convert a feature of intensity X_ij [0,1] into a latency coded response 

# a = torch.Tensor([0.02, 0.5, 0.5])

# print(a)

# a_data = spikegen.latency(a, num_steps = 5, normalize=True, linear=True)

# print(a_data)

# print(data_it)
# print(data_it.dim())
# print(data_it.size())


# Print first 10 key-value pairs for verification
# for key, value in list(pixel_dict.items()):
#     print(f"{key}: {value}\n")

# print(type(test_img_spike))
# print(test_img_spike.size())
# print(test_img_spike.dim())

# test_img = test_img.numpy()
# image_array_int = (test_img * 255).astype(int)

# for row in image_array_int:
#     print(" ".join(f"{pixel:3}" for pixel in row))  # Align numbers neatly

# plt.imshow(test_img, cmap="gray")  # Convert to NumPy and plot
# plt.title(f"Label: {targets_it[0].item()}")  # Display the corresponding label
# plt.axis("off")  # Hide axes
# plt.show()
