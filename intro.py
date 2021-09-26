import cv2

# imread
img = cv2.imread('dog.jpg')
# we donot want to destroy image so loop will be done.

while True:
    cv2.imshow("Dogimage", img)

    if cv2.waitKey(0) == 27:
        # we can use any key use ord('q')
        break
cv2.imwrite("saved/img.jpg",img)
cv2.destroyAllWindows()
