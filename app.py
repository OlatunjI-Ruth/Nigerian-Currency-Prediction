
  path='/content/' + fn
  img=image.load_img(path, target_size=(300, 300))
  x = image.img_to_array(img)
  x=x/225
  x= np.expand_dims(x, axis=0)
  image_tensor = np.vstack([x])
  classes = model.predict(image_tensor)
  classes = np.argmax(classes, axis=1)
  print(classes)
  if classes==2:
    print('This is 50 naira note')
  elif classes==0:
    print('This is a 1000 naira note') 
  elif classes==7:
    print('This is a 500 naira note')
  elif classes==4:
    print('This is a 200 naira note')
  elif classes == 5:
    print("This is a 5 naira note")
  elif classes==3:
    print('This is a 20 naira note')
  elif classes==6:
    print('This is a 100 Naira note')
  else:
    print('This ia a 10 Naira note')
