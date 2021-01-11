import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    image=cv2.imread('cameraman.tif')
    grayCameraman=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # Resmi siyah beyaz hale getirme

    img=cv2.imread('lena (1).bmp')
    grayLenaBmp=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    print('Cameraman Resim boyutu',str(image.shape))
    print('Cameraman Resim boyutu',str(grayCameraman.shape))

    imgshape=(image.shape)
    *_,color=imgshape

    if color==3:
        print('Cameraman Gorsel renk sayisi :',color, '. Gorsel renklidir.(BGR)')

    print('Resim toplam pixel degeri',str(image.size))
    print('Resim veri tipi',image.dtype) # islem yapabilmek icin once veri tipinin aynı olması gerek. bunu ogrenıyoruz


    while True:
        contrast=float(input('Enter a Contrast Value between 1 and 3 : '))
        if contrast<1:
            print('Enter a value in the given value')
            continue
        elif contrast>3:
            print('Enter a value in the given value.')
            continue
        else :
            print('The value entered is in the given range.')
            break

    while True :
        brigtness=float(input('Enter a Brightness Value between 0 and 100 : '))
        if brigtness<0:
            print('Enter a value in the given value')
            continue
        elif brigtness>100:
            print('Enter a value in the given value.')
            continue
        else :
            print('The value entered is in the given range.')
            break

    alpha=contrast
    beta=brigtness
    adjusted=cv2.convertScaleAbs(image,alpha=alpha,beta=beta)  # g(i,j)=alpha*f(i,j)+beta



    cv2.imshow('Cameraman',image)
    cv2.imshow('GrayCameraman',grayCameraman)
    cv2.imshow('AdjustedCameraman',adjusted)


    Z= grayCameraman.reshape((-1,1))
    Z= np.float32(Z)

    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1)

    K=1
    ret,label1,center1=cv2.kmeans(Z, K, None, criteria,10, cv2.KMEANS_RANDOM_CENTERS)
    center1=np.uint8(center1)
    res1=center1[label1.flatten()]
    output1 =res1.reshape((grayCameraman.shape))

    K = 2
    ret, label1, center1 = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center1 = np.uint8(center1)
    res1 = center1[label1.flatten()]
    output2 = res1.reshape((grayCameraman.shape))

    K = 3
    ret, label1, center1 = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center1 = np.uint8(center1)
    res1 = center1[label1.flatten()]
    output3 = res1.reshape((grayCameraman.shape))

    K = 4
    ret, label1, center1 = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center1 = np.uint8(center1)
    res1 = center1[label1.flatten()]
    output4 = res1.reshape((grayCameraman.shape))

    K = 5
    ret, label1, center1 = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center1 = np.uint8(center1)
    res1 = center1[label1.flatten()]
    output5 = res1.reshape((grayCameraman.shape))

    output=[image,output1,output2,output3,output4,output5]
    titles=['Lenna','K=1','K=2','K=3','K=4','K=5']


    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.imshow(output[i])
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()


    plt.hist(grayCameraman.ravel(), 256, [0, 256])
    plt.show()

    """BMP Formati PNG formatina gore daha ayrik degerlere sahip. 
       PNG Formati Histogram sonucu daha continious ve dar cikti."""





    cv2.waitKey(0) # Resmi göstermesi icin zamanlama
    cv2.destroyAllWindows() # Herhangi bir tusa basildiginda kapama


if __name__=='__main__' :
    main()