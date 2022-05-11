import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-chatbot',
  templateUrl: './chatbot.component.html',
  styleUrls: ['./chatbot.component.css']
})
export class ChatbotComponent implements OnInit {

  constructor() { }

  ngOnInit(): void {
  }

  ngAfterViewInit(): void{
    import('../../assets/js/RasaWidget.js');
  } 

  clickEvent(){
    console.log("You clicked!");
    let element :HTMLElement = document.getElementsByClassName("css-by5ua0")[0] as HTMLElement;
    element.click();
  }

}
