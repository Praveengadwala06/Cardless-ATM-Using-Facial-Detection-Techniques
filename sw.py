import csv
import os
import numpy as np
import imutils
import pickle
import cv2
from imutils import paths
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from imutils.video import VideoStream
from imutils.video import FPS
import time
from tkinter import *
from tkinter import messagebox
import pandas as pd

ARIAL = ("arial", 10, "bold")

class BankUi:
    def __init__(self, root):
        self.root = root
        self.header = Label(self.root, text="Automated Teller Machine (ATM)", bg="#0019fc", fg="white", font=("arial", 20, "bold"))
        self.header.pack(fill=X)
        self.frame = Frame(self.root, bg="#0019fc", width=900, height=500)
        root.geometry("800x500")
        self.button1 = Button(self.frame, text="Click to begin transactions", bg="#50A8B0", fg="white", font=ARIAL, command=self.begin_page)
        self.q = Button(self.frame, text="Quit", bg="#50A8B0", fg="white", font=ARIAL, command=self.root.destroy)
        self.q.place(x=340, y=340, width=200, height=40)
        self.button1.place(x=155, y=230, width=500, height=30)
        self.countter = 2
        self.frame.pack()

    def begin_page(self):
        self.frame.destroy()
        self.frame = Frame(self.root, bg="#0019fc", width=900, height=500)
        root.geometry("800x500")
        self.enroll = Button(self.frame, text="Enroll", bg="#50A8B0", fg="white", font=ARIAL, command=self.enroll_user)
        self.withdraw = Button(self.frame, text="Login", bg="#50A8B0", fg="white", font=ARIAL, command=self.withdraw_money_page)
        self.q = Button(self.frame, text="Quit", bg="#50A8B0", fg="white", font=ARIAL, command=self.root.destroy)
        self.enroll.place(x=0, y=315, width=200, height=50)
        self.withdraw.place(x=600, y=315, width=200, height=50)
        self.q.place(x=340, y=340, width=120, height=20)
        self.frame.pack()

    def withdraw_money_page(self):
        self.frame.destroy()
        self.frame = Frame(self.root, bg="#0019fc", width=900, height=500)
        self.label1 = Label(self.frame, text="Note:", bg="#0019fc", fg="white", font=ARIAL)
        self.label2 = Label(self.frame, text="1. By clicking on the 'Verify Face Id' button, we proceed to perform facial recognition.", bg="#0019fc", fg="white", font=ARIAL)
        self.label3 = Label(self.frame, text="2. Each capture will take 15 seconds and you are required to move your face in different directions while being captured.", bg="#0019fc", fg="white", font=ARIAL)
        self.label4 = Label(self.frame, text="3. If your face is recognized, you will be required to input your account password:", bg="#0019fc", fg="white", font=ARIAL)
        self.label5 = Label(self.frame, text="4. If your face is not recognized after 5 seconds, you will automatically be given 2 more trials.", bg="#0019fc", fg="white", font=ARIAL)
        self.label6 = Label(self.frame, text="5. If your face is not recognized after three trials, you won't be allowed to withdraw.", bg="#0019fc", fg="white", font=ARIAL)
        self.label7 = Label(self.frame, text="6. To begin, click the 'Verify Face Id' button below", bg="#0019fc", fg="white", font= ARIAL)
        self.button = Button(self.frame, text="Verify Face Id", bg="#50A8B0", fg="white", font=ARIAL, command=self.video_check)
        self.q = Button(self.frame, text="Quit", bg="#50A8B0", fg="white", font=ARIAL, command=self.root.destroy)
        self.b = Button(self.frame, text="Back", bg="#50A8B0", fg="white", font=ARIAL, command=self.begin_page)
        self.label1.place(x=100, y=100, width=800, height=20)
        self.label2.place(x=100, y=120, width=800, height=20)
        self.label3.place(x=100, y=140, width=800, height=20)
        self.label4.place(x=100, y=160, width=800, height=20 )
        self.label5.place(x=100, y=180, width=800, height=20)
        self.label6.place(x=100, y=200, width=800, height=20)
        self.label7.place(x=100, y=220, width=800, height=20)
        self.button.place(x=100, y=250, width=800, height=30)
        self.q.place(x=480, y=360, width=120, height=20)
        self.b.place(x=280, y=360, width=120, height=20)
        self.frame.pack()
        data = pd.read_csv('bank_details.csv')

    def enroll_user(self):
        self.frame.destroy()
        self.frame = Frame(self.root, bg="#0019fc", width=900, height=500)
        self.userlabel = Label(self.frame, text="Full Name", bg="#0019fc", fg="white", font=ARIAL)
        self.uentry = Entry(self.frame, bg="honeydew", highlightcolor="#50A8B0",
                            highlightthickness=2,
                            highlightbackground="white")
        self.plabel = Label(self.frame, text="Password", bg="#0019fc", fg="white", font=ARIAL)
        self.pentry = Entry(self.frame, bg="honeydew", show="*", highlightcolor="#50A8B0",
                            highlightthickness=2,
                            highlightbackground="white")
        self.button1 = Button(self.frame, text="Next", bg="#50A8B0", fg="white", font=ARIAL, command=self.enroll_and_move_to_next_screen)
        self.q = Button(self.frame, text="Quit", bg="#50A8B0", fg="white", font=ARIAL, command=self.root.destroy)
        self.b = Button(self.frame, text="Back", bg="#50A8B0", fg="white", font=ARIAL, command=self.begin_page)
        self.userlabel.place(x=125, y=100, width=120, height=20)
        self.uentry.place(x=153, y=130, width=200, height=20)
        self.plabel.place(x=125, y=160, width=120, height=20)
        self.pentry.place(x=153, y=190, width=200, height=20)
        self.button1.place(x=155, y=230, width=180, height=30)
        self.q.place(x=480, y=360, width=120, height=20)
        self.b.place(x=280, y=360, width=120, height=20)
        self.frame.pack()

    def enroll_and_move_to_next_screen(self):
        name = self.uentry.get()
        password = self.pentry.get()
        if not name and not password:
            messagebox._show("Error", "You need a name to enroll an account and you need to input a password!")
            self.enroll_user()
        elif not password:
            messagebox._show("Error", "You need to input a password!")
            self.enroll_user()
        elif not name:
            messagebox._show("Error", "You need a name to enroll an account!")
            self.enroll_user()
        elif len(password) < 8:
            messagebox._show("Password Error", "Your password needs to be at least 8 digits!")
            self.enroll_user()
        else:
            self.write_to_csv()
            self.video_capture_page()

    def password_verification(self):
        self.frame.destroy()
        self.frame = Frame(self.root, bg="#0019fc", width=900, height=500)
        print(self.real_user)
        self.plabel = Label(self.frame, text="Please enter your account password", bg="#0019fc", fg="white", font=ARIAL)
        self.givenpentry = Entry(self.frame, bg="honeydew", show="*", highlightcolor="#50A8B0",
                                 highlightthickness=2,
                                 highlightbackground="white")
        self.button1 = Button(self.frame, text="Verify", bg="#50A8B0", fg="white", font=ARIAL, command=self.verify_user)
        self.q = Button(self.frame, text="Quit", bg="#50A8B0", fg="white", font=ARIAL, command=self.root.destroy)
        self.b = Button(self.frame, text="Back", bg="#50A8B0", fg="white", font=ARIAL, command=self.begin_page)
        self.plabel.place(x=125, y=160, width=300, height=20)
        self.givenpentry.place(x=153, y=190, width=200, height=20)
        self.button1.place(x=155, y=230, width=180, height=30)
        self.q.place(x=480, y= 360, width=120, height=20)
        self.b.place(x=280, y=360, width=120, height=20)
        self.frame.pack()

    def verify_user(self):
        data = pd.read_csv('bank_details.csv')
        if data[data.loc[:, 'unique_id'] == self.real_user].empty:
            messagebox._show("Verification Info!", "User   not found")
            self.begin_page()
        else:
            self.gottenpassword = data[data.loc[:, 'unique_id'] == self.real_user].loc[:, 'password'].values[0]
            print(str(self.givenpentry.get()))
            print(str(self.gottenpassword))
            if str(self.givenpentry.get()) == str(self.gottenpassword):
                messagebox._show("Verification Info!", "Verification Successful!")
                self.final_page()
            else:
                messagebox._show("Verification Info!", "Verification Failed")
                self.begin_page()

    def final_page(self):
        self.frame.destroy()
        self.frame = Frame(self.root, bg="#0019fc", width=900, height=500)
        self.detail = Button(self.frame, text="Transfer", bg="#50A8B0", fg="white", font=ARIAL, command=self.user_account_transfer)
        self.enquiry = Button(self.frame, text="Balance Enquiry", bg="#50A8B0", fg="white", font=ARIAL, command=self.user_balance)
        self.deposit = Button(self.frame, text="Deposit Money", bg="#50A8B0", fg="white", font=ARIAL, command=self.user_deposit_money)
        self.withdrawl = Button(self.frame, text="Withdrawl Money", bg="#50A8B0", fg="white", font=ARIAL, command=self.user_withdrawl_money)
        self.q = Button(self.frame, text="Log out", bg="#50A8B0", fg="white", font=ARIAL, command=self.begin_page)
        self.detail.place(x=0, y=0, width=200, height=50)
        self.enquiry.place(x=0, y=315, width=200, height=50)
        self.deposit.place(x=600, y=0, width=200, height=50)
        self.withdrawl.place(x=600, y=315, width=200, height=50)
        self.q.place(x=340, y=340, width=120, height=20)
        self.frame.pack()

    def user_account_transfer(self):
        self.frame.destroy()
        self.frame = Frame(self.root, bg="#0019fc", width=900, height=500)
        self.label11 = Label(self.frame, text="Please enter the recipient's account number", bg="#0019fc", fg="white", font=ARIAL)
        self.label21 = Label(self.frame, text="Please enter the amount to be transferred", bg="#0019fc", fg="white", font=ARIAL)
        self.button1 = Button(self.frame, text="Transfer", bg="#50A8B0", fg="white", font=ARIAL, command=self.user_account_transfer_transc)
        self.entry11 = Entry(self.frame, bg="honeydew", highlightcolor="#50A8B0",
                             highlightthickness=2,
                             highlightbackground="white")
        self.entry21 = Entry(self.frame, bg="honeydew", highlightcolor="#50A8B0",
                             highlightthickness=2,
                             highlightbackground="white")
        self.label11.place(x=125, y=100, width=300, height=20)
        self.entry11.place(x=153, y=130, width=200, height=20)
        self.label21.place(x=125, y=160, width=300, height=20)
        self.entry21.place(x=153, y=190, width=200, height=20)
        self.button1.place(x=155, y=230, width=180, height=30)
        self.q = Button(self.frame, text="Quit", bg="#50A8B0", fg="white", font=ARIAL, command=self.root.destroy)
        self.q.place(x=480, y=360, width=120, height=20)
        self.b = Button(self.frame, text="Back", bg="#50A8B0", fg="white", font=ARIAL, command=self.begin_page)
        self.b.place(x=280, y=360, width=120, height=20)
        self.frame.pack()

    def user_account_transfer_transc(self):
        data = pd.read_csv('bank_details.csv')
        recipient_account = int(self.entry11.get())
        transfer_amount = int(self.entry21.get())
        if recipient_account not in data['account_number'].values:
            messagebox.showerror("Transfer Info!", "Invalid account number ")
            return
        if recipient_account == data[data['unique_id'] == self.real_user]['account_number'].values[0]:
            messagebox.showerror("Transfer Info!", "You cannot transfer money to yourself")
            return
        user_balance = data[data['unique_id'] == self.real_user]['account_balance'].values[0]
        if transfer_amount > user_balance:
            messagebox.showerror("Transfer Info!", "Insufficient balance")
            return
        data.loc[data['account_number'] == recipient_account, 'account_balance'] += transfer_amount
        data.loc[data['unique_id'] == self.real_user, 'account_balance'] -= transfer_amount
        data.to_csv('bank_details.csv', index=False)
        messagebox.showinfo("Transfer Info!", "Transfer successful!")

    def user_balance(self):
        data = pd.read_csv('bank_details.csv')
        user_balance = data[data['unique_id'] == self.real_user]['account_balance'].values[0]
        if pd.isnull(user_balance):
            messagebox.showinfo("Balance Info!", "Your account balance is not initialized. Please deposit money first.")
        else:
            messagebox.showinfo("Balance Info!", f"Your current balance is: Rs.{user_balance}")

    def user_deposit_money(self):
        self.frame.destroy()
        self.frame = Frame(self.root, bg="#0019fc", width=900, height=500)
        self.label = Label(self.frame, text="Enter amount to deposit", font=ARIAL)
        self.label.place(x=200, y=100, width=300, height=100)
        self.money_box = Entry(self.frame, bg="honeydew", highlightcolor="#50A8B0",
                               highlightthickness=2,
                               highlightbackground="white")
        self.money_box.place(x=200, y=130, width=200, height=20)
        self.submitButton = Button(self.frame, text="Deposit", bg="#50A8B0", fg="white", font=ARIAL,
                                   command=self.user_deposit_trans)
        self.submitButton.place(x=445, y=130, width=55, height=20)
        self.q = Button(self.frame, text="Quit", bg="#50A8B0", fg="white", font=ARIAL, command=self.root.destroy)
        self.q.place(x=480, y=360, width=120, height=20)
        self.b = Button(self.frame, text="Back", bg="#50A8B0", fg="white", font=ARIAL, command=self.final_page)
        self.b.place(x=280, y=360, width=120, height=20)
        self.frame.pack()

    def user_deposit_trans(self):
        data = pd.read_csv('bank_details.csv')
        deposit_amount = int(self.money_box.get())
        if pd.isnull(data[data['unique_id'] == self.real_user]['account_balance'].values[0]):
            data.loc[data['unique_id'] == self.real_user, 'account_balance'] = deposit_amount
        else:
            data.loc[data['unique_id'] == self.real_user, 'account_balance'] += deposit_amount
        data.to_csv('bank_details.csv', index=False)
        messagebox.showinfo("Deposit Info!", "Deposit successful!")
        self.final_page()

    def user_withdrawl_money(self):
        self.frame.destroy()
        self.frame = Frame(self.root, bg="#0019fc", width=900, height=500)
        self.label = Label(self.frame, text="Enter amount to withdraw", font=ARIAL)
        self.label.place(x=200, y=100, width=300, height=100)
        self.money_box = Entry(self.frame, bg="honeydew", highlightcolor="#50A8B0",
                               highlightthickness=2,
                               highlightbackground="white")
        self.money_box.place(x=200, y=130, width=200, height=20)
        self.submitButton = Button(self.frame, text="Withdraw", bg="#50A8B0", fg="white", font=ARIAL,
                                   command=self.user_withdrawl_trans)
        self.submitButton.place(x=445, y=130, width=70, height=20)
        self.q = Button(self.frame, text="Quit", bg="#50A8B0", fg="white", font=ARIAL, command=self.root.destroy)
        self.q.place(x=480, y=360, width=120, height=20)
        self.b = Button(self.frame, text="Back", bg="#50A8B0", fg="white", font=ARIAL, command=self.final_page)
        self.b.place(x=280, y=360, width=120, height=20)
        self.frame.pack()

    def user_withdrawl_trans(self):
        data = pd.read_csv('bank_details.csv')
        withdraw_amount = int(self.money_box.get())
        user_balance = data[data['unique_id'] == self.real_user]['account_balance'].values[ 0]
        if withdraw_amount > user_balance:
            messagebox.showerror("Withdrawal Info!", "Insufficient funds")
            return
        data.loc[data['unique_id'] == self.real_user, 'account_balance'] -= withdraw_amount
        data.to_csv('bank_details.csv', index=False)
        messagebox.showinfo("Withdrawal Info!", "Withdrawal successful!")
        self.final_page()

    def write_to_csv(self):
        import csv
        from random import randint
        n = 10
        range_start = 10 ** (n - 1)
        range_end = (10 ** n) - 1
        account_number = randint(range_start, range_end)
        n = 5
        range_start = 10 ** (n - 1)
        range_end = (10 ** n) - 1
        unique_id = randint(range_start, range_end)
        bank = "Unilag Bank"
        account_balance = "10000"
        name = self.uentry.get()
        password = self.pentry.get()
        with open(r'bank_details.csv', 'a', newline='\n') as f:
            writer = csv.writer(f)
            writer.writerow([unique_id, account_number, name, bank, password, account_balance])
        messagebox._show("Enrollment Info!", "Successfully Enrolled!")

    def video_capture_page(self):
        self.frame.destroy()
        self.frame = Frame(self.root, bg="#0019fc", width=900, height=500)
        self.label = Label(self.frame, text="Please wait while we capture your face", bg="#0019fc", fg="white", font=ARIAL)
        self.button = Button(self.frame, text="Capture Face", bg="#50A8B0", fg="white", font=ARIAL, command=self.captureuser)
        self.q = Button(self.frame, text="Quit", bg="#50A8B0", fg="white", font=ARIAL, command=self.root.destroy)
        self.b = Button(self.frame, text="Back", bg="#50A8B0", fg="white", font=ARIAL, command=self.begin_page)
        self.label.place(x=100, y=160, width=600, height=20)
        self.button.place(x=100, y=230, width=600, height=30)
        self.q.place(x=480, y=360, width=120, height=20)
        self.b.place(x=280, y=360, width=120, height=20)
        self.frame.pack()

    def captureuser(self):
        data = pd.read_csv('bank_details.csv')
        name = data.loc[:, 'unique_id'].values[-1]
        cam = cv2.VideoCapture(0)

        cv2.namedWindow("capture")

        img_counter = 0

        dirname = f'dataset/{name}'
        os.mkdir(dirname)

        while True:
            ret, frame = cam.read()
            cv2.imshow("capture", frame)

            if img_counter == 5:
                cv2.destroyWindow("capture")
                break
            if not ret:
                break
            k = cv2.waitKey(1)

            if k % 256 == 27:
                print("Escape hit, closing...")
                break
            elif k % 256 == 32:
                path = f'dataset/{name}'
                img_name = "{}.jpg".format(img_counter)
                cv2.imwrite(os.path.join(path, img_name), frame)
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
                img_counter += 1

        cam.release()

        cv2.destroyAllWindows()

        self.get_embeddings()
        self.train_model()
        messagebox._show("Registration Info!", "Face Id Successfully Registered!")
        self.begin_page()

    def get_embeddings(self):
        print("[INFO] quantifying faces...")

        detector = cv2.dnn.readNetFromCaffe('face_detection_model/deploy.prototxt', 'face_detection_model/res10_300x300_ssd_iter_140000.caffemodel')
        embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')

        imagePaths = list(paths.list_images('dataset'))
        knownEmbeddings = []
        knownNames = []
        total = 0

        for (i, imagePath) in enumerate(imagePaths):
            print(f"[INFO] processing image {i + 1}/{len(imagePaths)}")
            name = imagePath.split(os.path.sep)[-2]

            image = cv2.imread(imagePath)
            image = imutils.resize(image, width=600)
            (h, w) = image.shape[:2]

            imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
                                            (104.0, 177.0, 123.0), swapRB=False, crop =False)

            detector.setInput(imageBlob)
            detections = detector.forward()

            if len(detections) > 0:
                i = np.argmax(detections[0, 0, :, 2])
                confidence = detections[0, 0, i, 2]

                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    face = image[startY:endY, startX:endX]
                    (fH, fW) = face.shape[:2]

                    if fW < 20 or fH < 20:
                        continue

                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                                    (0, 0, 0), swapRB=True, crop=False)
                    embedder.setInput(faceBlob)
                    vec = embedder.forward()

                    knownNames.append(name)
                    knownEmbeddings.append(vec.flatten())
                    total += 1

        print(f"[INFO] serializing {total} encodings...")
        data = {"embeddings": knownEmbeddings, "names": knownNames}
        f = open('output/embeddings.pickle', "wb")
        f.write(pickle.dumps(data))
        f.close()

    def train_model(self):
        print("[INFO] loading face embeddings...")
        data = pickle.loads(open('output/embeddings.pickle', "rb").read())
        le = LabelEncoder()
        labels = le.fit_transform(data["names"])
        recognizer = SVC(C=1.0, kernel="linear", probability=True)
        recognizer.fit(data["embeddings"], labels)
        f = open('output/recognizer.pickle', "wb")
        f.write(pickle.dumps(recognizer))
        f.close()

        f = open('output/le.pickle', "wb")
        f.write(pickle.dumps(le))
        f.close()

    def video_check(self):
        detector = cv2.dnn.readNetFromCaffe('face_detection_model/deploy.prototxt', 'face_detection_model/res10_300x300_ssd_iter_140000.caffemodel')
        embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')

        recognizer = pickle.loads(open('output/recognizer.pickle', "rb").read())
        le = pickle.loads(open('output/le.pickle', "rb").read())

        vs = VideoStream(src=0).start()
        time.sleep(2.0)

        timeout = time.time() + 5

        fps = FPS().start()

        real_user_list = []
        while True:
            if time.time() > timeout:
                cv2.destroyWindow("Frame")
                break

            frame = vs.read()

            frame = imutils.resize(frame, width=800, height=200)
            (h, w) = frame.shape[:2]

            imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                                            (104.0, 177.0, 123.0), swapRB=False, crop=False)

            detector.setInput(imageBlob)
            detections = detector.forward()

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    face = frame[startY:endY, startX:endX]
                    (fH, fW) = face.shape[:2]

                    if fW < 20 or fH < 20:
                        continue

                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                                    (0, 0, 0), swapRB=True, crop=False)
                    embedder.setInput(faceBlob)
                    vec = embedder.forward()

                    preds = recognizer.predict_proba(vec)[0]
                    j = np.argmax(preds)
                    proba = preds[j]
                    name = le.classes_[j ]

                    if (name == 'unknown') or (proba * 100) < 50:
                        print("detected")
                        real_user_list.append(name)
                    else:
                        real_user_list.append(name)
                        break

            fps.update()

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        cv2.destroyAllWindows()
        vs.stop()
        print(real_user_list)

        try:
            Counter(real_user_list).most_common(1)[0][0] == 'unknown'
        except IndexError:
            if self.countter != 0:
                messagebox._show("Verification Info!", "Face Id match failed! You have {} trials left".format(self.countter))
                self.countter = self.countter - 1
                self.video_check()
            else:
                messagebox._show("Verification Info!", "Face Id match failed! You cannot withdraw at this time, try again later")
                self.begin_page()
                self.countter = 2

        else:
            if Counter(real_user_list).most_common(1)[0][0] == 'unknown':
                if self.countter != 0:
                    messagebox._show("Verification Info!", "Face Id match failed! You have {} trials left".format(self.countter))
                    self.countter = self.countter - 1
                    self.video_check()
                else:
                    messagebox._show("Verification Info!", "Face Id match failed! You cannot withdraw at this time, try again later")
                    self.begin_page()
                    self.countter = 2

            else:
                self.real_user = int(Counter(real_user_list).most_common(1)[0][0])
                messagebox._show("Verification Info!", "Face Id match!")
                self.password_verification()

root = Tk()
root.title("Automated Teller Machine (ATM)")
root.geometry("800x500")
root.configure(bg="blue")
obj = BankUi(root)
root.mainloop()