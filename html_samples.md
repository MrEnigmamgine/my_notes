<div markdown="1" 
    style="
    box-sizing: border-box;
    background-color: #f26200; 
    color: white; 
    border-radius: 15px; 
    padding: 15px; 
    width: 55%">
 <strong>Caution!</strong> <br>
  Models trained on biased data will reflect the bias of the humans that labeled it.  <br>
  Because this data was created by a small group, there is a high risk of bias.
</div>



<div markdown="1" 
    style= "
    box-sizing: border-box;
    padding: 15px;
    margin-bottom: 20px;
    border: 1px solid transparent;
    border-radius: 15px;
    color: #31708f;
    background-color: #d9edf7;
    border-color: #bce8f1;
    width: 55%;
    ">

**Heads Up!**  
The order of the results of a query are not implicitly ordered, so if you're paging, make sure you provide an [`$order` clause](https://dev.socrata.com/docs/queries/) or at a minimum `$order=:id`.  
That will guarantee that the order of your results will be stable as you page through the dataset.
</div>