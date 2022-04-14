import {Component, OnInit} from '@angular/core';
declare function addChat(): void;
@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit{
  title = 'frontend';
  constructor() {
  }

  ngOnInit() : void{
    addChat();
  }


}
