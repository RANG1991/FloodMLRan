<!-- Intro-->

<!--
* Thanks for reviewing my Project-README-Template! 
* 
* Read the comments for an easy step by step guide. Enjoy!
-->

<!-- Logo Section  --> <!-- Required -->

<!--
* Insert your github profile URL in the <a> "href" attribute bellow (line-25)
* 
* Insert an image URL in the <img> "src" attribute bellow. (line-26)
-->
<div align="center">
    <a href="https://github.com/RANG1991" target="_blank">
    </a>
</div>


<!-- Project title 
* use a dynamic typing-SvG here https://readme-typing-svg.demolab.com/demo/
*
*  Instead you can type your project name after a # header
-->

<div align="center">
    <h1>Deep Learning For Regional Spatio-Temporal Streamflow Prediction</h1>
</div>

<div align="center">
    <h2>About</h2>
</div>
<!-- 
* information about the project 
* 
* keep it short and sweet
-->
<div align="center">
  In this thesis project, we intestigated the use of CNN-LSTM and CNN-Transofrmer architectures to predict the amount 
  of discharge (the volume of water running through a cross section in va single day) in multiple basins around the US. 
  We used 32 features (5 dynamic and 27 static) as the inputs to our 2 models, and predicted a single scalar value (the discharge) as the output.
</div>

</br>
<p align="center">
  <em>Commulative density functions for our 2 models (CNN LSTM and CNN Trasnformer) and the baseline model (LSTM) for the test period.
  Graphs that are closer to the lower right corner perform better.</em>
  <img src="https://github.com/RANG1991/FloodMLRan/blob/main/static/images/NSE_CDF_CNN_LSTM_CNN_Transformer_LSTM_test.png" 
  alt="NSE_CDF_CNN_LSTM_CNN_Transformer_LSTM_test" 
  style="width:700px;height:500px;">
</p>

</br>
<p align="center">
  <em>Commulative density functions for our 2 models (CNN LSTM and CNN Trasnformer) and the baseline model (LSTM) for the validation period.
  Graphs that are closer to the lower right corner perform better.</em>
  <img src="https://github.com/RANG1991/FloodMLRan/blob/main/static/images/NSE_CDF_CNN_LSTM_CNN_Transformer_LSTM_validation.png" 
  alt="NSE_CDF_CNN_LSTM_CNN_Transformer_LSTM_test" 
  style="width:700px;height:500px;">
</p>

</br>
<p align="center">
  <em>The pipeline of out first model (CNN-LSTM). We used 2 LSTMs, where the first one leverages only the non-spatial inputs, and the second one leverages also the precipitation maps.</em>
  <img src="https://github.com/RANG1991/FloodMLRan/blob/main/static/images/Slide3.PNG" 
  alt="NSE_CDF_CNN_LSTM_CNN_Transformer_LSTM_test" 
  style="width:800px;height:500px;">
</p>

</br>
<p align="center">
  <em> The pipeline of out second model (CNN-Transformer). We used 2 Transformer encoders, where the first one leverages only the non-spatial inputs, and the second one leverages also the precipitation maps. 
    The second encoder uses cross attention ro encorporate the output of the first encoder with the output of the CNN.</em>
  <img src="https://github.com/RANG1991/FloodMLRan/blob/main/static/images/Slide6.PNG" 
  alt="NSE_CDF_CNN_LSTM_CNN_Transformer_LSTM_test" 
  style="width:800px;height:500px;">
</p>
