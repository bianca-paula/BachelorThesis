import {Component, OnInit} from '@angular/core';
// declare function addChat(): void;
@Component({
  selector: 'app-home',
  templateUrl: 'home.page.html',
  styleUrls: ['home.page.scss'],
})
export class HomePage implements OnInit{

  constructor() {}

  ngOnInit(): void{
    // addChat();
  }

  ngAfterViewInit(): void{
    import('../../assets/js/RasaWidget.js');
  } 
}
