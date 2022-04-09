import { NgModule } from '@angular/core';
import { RouteReuseStrategy } from '@angular/router';

// import {ChatbotRasaModule} from 'angular-chat-widget-rasa';
import { IonicModule, IonicRouteStrategy } from '@ionic/angular';
import { AppComponent } from './app.component';
import { AppRoutingModule } from './app-routing.module';
import {BrowserModule} from '@angular/platform-browser';


@NgModule({
  declarations: [AppComponent],
  entryComponents: [],
  imports: [BrowserModule, IonicModule.forRoot(), AppRoutingModule],
  providers: [{ provide: RouteReuseStrategy, useClass: IonicRouteStrategy }],
  bootstrap: [AppComponent],
})
export class AppModule {}
