path = 'dataset/test_set/dogs/dog.4001.jpg'
img = imread(path)
plt.imshow(img)
plt.show()
img_resize = resize(img, (150, 150, 3))
 l = [img_resize.flatten()]
  probability = model.predict_proba(l)
   for ind, val in enumerate(Categories):
       print(f'{val} = {probability[0][ind] * 100}%')
   print("The predicted image is : " + Categories[model.predict(l)[0]])
   