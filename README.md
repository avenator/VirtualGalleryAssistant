# Virtual Gallery Assistant

## Реализация итогового проекта "Цифровой музей" в Deep Learning School от МФТИ

- Разработка tg-бота для того, чтобы сделать выставки в музеях более интерактивными.

## Интерфейс
<img src="https://github.com/avenator/VirtualGalleryAssistant/blob/main/pics/example.png" height="500" width="300">

- Style Tranfer выполен с использованием fine - tuned версии [JoJoGan](https://github.com/mchong6/JoJoGAN)
- Замена лица была выполнена с помощью [FaceSwap](https://github.com/wuhuikai/FaceSwap/tree/master)  
- Веса моделей, необходимые для запуска бота на [диске](https://drive.google.com/drive/folders/1ex0Ixlh2yc92T3nyrahBRbJai9VlbTGS?usp=sharing)
## Style Transfer (alpha=1)

Исходная фотография        |  Картина                  |  Стилизованная фотография |  Итоговая фотография
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://github.com/avenator/VirtualGalleryAssistant/blob/main/pics/depp.jpg" height="250" width="250">  |  <img src="https://github.com/avenator/VirtualGalleryAssistant/blob/main/pics/Gogh.jpg" height="250" width="250">  |   <img src="https://github.com/avenator/VirtualGalleryAssistant/blob/main/pics/gogh_edit.jpg" height="250" width="250"> | <img src="https://github.com/avenator/VirtualGalleryAssistant/blob/main/pics/gogh_fianl.jpg" height="250" width="250"> 
<img src="https://github.com/avenator/VirtualGalleryAssistant/blob/main/pics/zendaya.jpg" height="250" width="250">  |  <img src="https://github.com/avenator/VirtualGalleryAssistant/blob/main/pics/Mona-Lisa.jpg" height="250" width="250">  |   <img src="https://github.com/avenator/VirtualGalleryAssistant/blob/main/pics/lisa_edit.jpg" height="250" width="250">| <img src="https://github.com/avenator/VirtualGalleryAssistant/blob/main/pics/lisa_fianl.jpg" height="250" width="250"> 
<img src="https://github.com/avenator/VirtualGalleryAssistant/blob/main/pics/jackman.jpg" height="250" width="250">  |  <img src="https://github.com/avenator/VirtualGalleryAssistant/blob/main/pics/Beethoven.jpg" height="250" width="250">  |   <img src="https://github.com/avenator/VirtualGalleryAssistant/blob/main/pics/beethoven_edit.jpg" height="250" width="250"> | <img src="https://github.com/avenator/VirtualGalleryAssistant/blob/main/pics/beethoven_fianl.jpg" height="250" width="250"> 
<img src="https://github.com/avenator/VirtualGalleryAssistant/blob/main/pics/lawrence.jpg" height="250" width="250">  |  <img src="https://github.com/avenator/VirtualGalleryAssistant/blob/main/pics/monro.jpg" height="250" width="250">  |   <img src="https://github.com/avenator/VirtualGalleryAssistant/blob/main/pics/monro_edit.jpg" height="250" width="250"> | <img src="https://github.com/avenator/VirtualGalleryAssistant/blob/main/pics/monro_fianl.jpg" height="250" width="250"> 
