import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import 'rxjs/Observable/of';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { ChatbotRasaModule } from 'angular-chat-widget-rasa';

@NgModule({
  declarations: [
    AppComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    ChatbotRasaModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
