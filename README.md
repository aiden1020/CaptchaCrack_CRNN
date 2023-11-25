# CaptchaCrack_CRNN
# Captcha CRNN Model Build + CTC Loss

Tags: CNN, Captcha Cracker, LSTM, Pytorch

## 為什麼不能直接用CNN 辨識字串?

- 對於具有多個數字的圖像，例如四個數字的排列，可能有許多不同的組合。這樣的情況使得使用傳統的CNN 可能難以直接捕捉所有可能的排列，因為模型需要學習大量的變化。
- 此外，數字的位置、大小和方向的變化也會增加難度。

## CRNN架構
![Untitled](https://github.com/aiden1020/CaptchaCrack_CRNN/assets/127687126/ff6d8f40-568c-4f87-818e-9af9dcde120d)


**CNN Layer（卷積層）**

- 用於提取圖像的局部特徵。
- 輸出Convolutional feature maps（卷積特徵圖）。

**Map to sequence（映射到序列）**

- 將Convolutional feature maps轉換為feature sequence，以便能夠將圖像的特徵序列輸入到RNN中學習時序信息。

**RNN Layer（遞歸層）**

- 使用Feature sequence學習圖像中由左至右的時序信息，這對於文字識別中的序列性資訊非常重要。

CNN主要負責提取圖像的區域特徵，而RNN則在整個特徵序列中學習時序信息，以便更好地理解和識別圖像中的文字。這種結合CNN和RNN的方法，特別是在文字識別（OCR）等任務中，可以有效處理圖像中的序列性信息，提高模型的性能。

## CNN Layers
![Untitled 2](https://github.com/aiden1020/CaptchaCrack_CRNN/assets/127687126/1fdd716d-21fd-4caa-972a-8040d39f4560)
![Untitled 1](https://github.com/aiden1020/CaptchaCrack_CRNN/assets/127687126/2a4bdf7a-79ff-49db-992f-d6d872e70e30)


- 輸入圖像格式為(N, C , H , W)
- 使用Conv 層和MaxPooling層 Downsample
- 作者在maxpooling2 ,maxpooling3 使用不對稱的kernel filter (2,1) 長條形的filter 有利於捕捉文字類特徵
- 輸出Conv Feature Maps (N ,512,1,25) 就是512層 1 x 25特徵圖
    
    N : Batch size
    
    C : Channel
    
    H : Height
    
    W : Width
    

### Map to Sequence
![Untitled 3](https://github.com/aiden1020/CaptchaCrack_CRNN/assets/127687126/bec959a2-29c9-4735-af46-63dd074a8b34)


- 把Conv Feature Maps (N, C , H , W) 轉換成LSTM 能接受的形狀 $(L,N,$$H_{in}$)
- 所以 (N ,512,1,25) 轉換成 (25,N,512)

L : Sequence Length(time step)

N : Batch Size

$H_{in}$ : Input Size(feature number)

## RNN Layers
![Untitled 4](https://github.com/aiden1020/CaptchaCrack_CRNN/assets/127687126/2aca827b-8381-483b-868e-1dcb779ad67b)


- 使用Bidirectional LSTM (雙向LSTM)原因是可以同時處理正向（左到右）和反向（右到左）的序列，捕捉字符之間的上下文信息
- Bidirectional LSTM 輸入形狀為 $(L,N,$$H_{in}$)  輸出形狀為 $(N,L,2∗H_{cell})$
    
    L : Sequence Length(time step)
    
    N : Batch Size
    
    $H_{in}$ : Input Size(feature number)
    
    $H_{cell}$  : Hidden Size
    

## Transcription Layers

![Untitled 5](https://github.com/aiden1020/CaptchaCrack_CRNN/assets/127687126/86937784-b0a2-4439-97d0-ff0b14013563)


- 轉錄是將RNN每幀（frame）的預測轉換為標籤序列的過程
- 在數學上，轉錄的目標是在每幀的預測條件下找到概率最高的標籤序列

### Connectionist Temporal Classification (CTC) layer

- Sequence(RNN output) to Sequence(target label) 的模型對齊是十分困難，

## Reference

- Reference
    
    [OCR-驗證碼識別理論與實作](https://cinnamonaitaiwan.medium.com/ocr-驗證碼識別理論與實作-a97273a5657d)
    
    - Shi, B., Bai, X., & Yao, C. (2015, July 21). *An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition*. arXiv.org.
        
         [https://arxiv.org/abs/1507.05717](https://arxiv.org/abs/1507.05717) 
        
        [https://arxiv.org/pdf/1507.05717.pdf](https://arxiv.org/pdf/1507.05717.pdf)
        

---
