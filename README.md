基於CNN技術的人體部位圖片識別
===========================
### 使用CNN技術辨識圖片中是人體的耳朵、手或是腳  
#### 此專案是暑假課程時，與小組成員共同完成的專案，我們從蒐集資料集照片開始一步一步到編寫照片預測程式碼  
不過在此篇github只上傳資料集製作、訓練模型以及測試資料預測的程式碼 
* 完整專案連結(包含圖片、標註檔案等，檔案大小約5.5GB)：https://drive.google.com/file/d/1cbgIfuUpIf54KX2WGYxkhVAF_gE2U54z/view?usp=sharing  

專案流程：
---------
1.蒐集專案資料照片  
2.由於大家的照片名稱不一樣，需先將照片統一檔名，好方便後續讀取照片  
3.將圖片檔案標記整理成CSV檔案  
4.將訓練照片與標記資料製作成訓練資料集NPY檔  
5.將資料集進行預處理後進行模型訓練，並將訓練好的模型權重儲存在h5檔案裡  
6.將未訓練的測試資料透過訓練好的模型預測，評估其模型正確率  

### 模型摘要：
![image](https://github.com/Lily-Liao/CNN-Project-Body-Identify-/blob/main/model.png)
### 訓練過程：
* accuracy執行結果:  
![image](https://github.com/Lily-Liao/CNN-Project-Body-Identify-/blob/main/accuracy.png)
* loss誤差執行結果：  
![image](https://github.com/Lily-Liao/CNN-Project-Body-Identify-/blob/main/loss.png)
### 驗證模型準確率：  
![image](https://github.com/Lily-Liao/CNN-Project-Body-Identify-/blob/main/test%20scores.png)

檢討：
-----
根據訓練過程的驗證曲線可以看出其非常不平穩以及測試資料驗證模型準確率不夠高。  
* 解決方法一：  
>可能是資料集問題，可蒐集多樣性的照片進行訓練，或是將照片進行輪廓檢測去掉其他因素的影響  
* 解決方法二：  
>可調整看看learning rate使其來回震動不會那麼大
