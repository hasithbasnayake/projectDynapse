import torch
from sklearn.linear_model import LinearRegression
from snntorch import spikegen
import matplotlib.pyplot as plt
from model import *

test_net = Net(num_input=1568, num_output=2, beta=0.8, threshold=20, reset_mechanism="zero")
test_net.load_state_dict(torch.load('test_net.pt'))

data_ON = torch.load("data/processed/convON_train.pt", weights_only=True) 
data_OFF = torch.load("data/processed/convOFF_train.pt", weights_only=True) 

test_images_ON = data_ON[:1000]
test_images_OFF = data_OFF[:1000]

X = []
Y = []

for iter in range(200):
    img_ON, label_ON = test_images_ON[iter]
    img_OFF, label_OFF = test_images_OFF[iter]

    img = torch.stack([img_ON, img_OFF], dim=1)

    img = torch.flatten(img, start_dim=1)
    Y.append(img)
    img = spikegen.latency(data=img, num_steps=255, normalize=True, linear=True)
    img = img.squeeze(1)


    spk_rec, mem_rec = test_net(img)

    mean_spk = torch.sum(spk_rec, dim=0)
    mean_spk = mean_spk / 255

    # print(f"mean_spk: {mean_spk.shape}")
    # print(mean_spk)

    X.append(mean_spk)

X = torch.stack(X)
Y = torch.stack(Y)
Y = Y.squeeze(1)

print(X.shape)
print(Y.squeeze(1).shape)

decoder = LinearRegression()
decoder.fit(X.detach().numpy(), Y.detach().numpy())

img_ON, label_ON = test_images_ON[0]
img_OFF, label_OFF = test_images_OFF[0]
img = torch.stack([img_ON, img_OFF], dim=1)
img = torch.flatten(img, start_dim=1)

print(img.shape)

spk_rec, mem_rec = test_net(img)
mean_spk = torch.sum(spk_rec, dim=0)
mean_spk = mean_spk / 255

print(mean_spk.shape)

r_img = decoder.predict(mean_spk.unsqueeze(0).detach().numpy())

r_img = torch.from_numpy(r_img.squeeze(0))

print(r_img.shape)

rf_ON = r_img[:784].reshape(28, 28)
rf_OFF = r_img[784:].reshape(28, 28)

fig, axes = plt.subplots(1, 4, figsize=(15, 5))

# Plot the first image ON
axes[0].imshow(img_ON.squeeze().cpu().numpy(), cmap='gray')
axes[0].set_title('Original ON Image')
axes[0].axis('off')

axes[1].imshow(img_OFF.squeeze().cpu().numpy(), cmap='gray')
axes[1].set_title('Original OFF Image')
axes[1].axis('off')

# Plot the reconstructed rf_ON image
axes[2].imshow(rf_ON.cpu().numpy(), cmap='gray')
axes[2].set_title('Reconstructed rf_ON')
axes[2].axis('off')

# Plot the reconstructed rf_OFF image
axes[3].imshow(rf_OFF.cpu().numpy(), cmap='gray')
axes[3].set_title('Reconstructed rf_OFF')
axes[3].axis('off')

# Display the plot
plt.tight_layout()
plt.show()
