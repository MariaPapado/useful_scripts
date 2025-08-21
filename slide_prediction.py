


def find_dxs_dys(width, height, psize, step):
    img_x, img_y = [], []

    for x in range(0, height, step):
        if x+psize<=height:
            img_x.append(x)
        else:
            img_x.append(height-psize)
            break
        if x==0:
            for y in range(0, width, step):
                if y+psize<=width:
                    img_y.append(y)
                else:
                    img_y.append(width-psize)
                    break
    return img_x, img_y

img = Image.open('/home/maria/DATASETS/openearthmap/test/images_png/{}'.format(id))
img = np.array(img)

nb_classes = 9
psize = 256
step=32

votes = np.zeros(img.shape[:2] + (nb_classes,))
img_x, img_y = find_dxs_dys(img.shape[0], img.shape[1], psize, step)
img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().cuda()


for x in img_x:
    for y in img_y:
        img_patch = img[:,:,x:x+psize,y:y+psize]     
        prediction = F.softmax(model(img_patch), 1)
        prediction = prediction.squeeze().permute(1,2,0).data.cpu().numpy()
        votes[x:x+psize,y:y+psize] = votes[x:x+psize,y:y+psize] + prediction

predictions = np.argmax(votes, 2) #(512,512)
