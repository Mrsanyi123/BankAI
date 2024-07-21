const toggleSwitch = document.getElementById("toggle-btn");
const annualBasic = document.querySelector(".basic-pricing")
const annualMaster = document.querySelector(".master-pricing")

toggleSwitch.addEventListener("click", function(){
    if(this.checked){
        annualBasic.textContent = "$0"
        annualMaster.textContent = "$114.99"
    }
    else{
        annualBasic.textContent = "$0"
        annualMaster.textContent = "$11.99"
    }
})