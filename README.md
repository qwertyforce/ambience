# ambience
reverse image search and similarity search engine <br>
[How it works.md](https://github.com/qwertyforce/ambience/blob/main/how_it_works_search.md)  
API Gateway for: 
- (global_features_web - https://github.com/qwertyforce/global_features_web)
- (local_features_web - https://github.com/qwertyforce/local_features_web)
- (phash_web - https://github.com/qwertyforce/phash_web)
- (color_web - https://github.com/qwertyforce/color_web)
- (image_text_features_web - https://github.com/qwertyforce/image_text_features_web)
- (image_caption_web - https://github.com/qwertyforce/image_caption_web)
- (places365_tagger_web - https://github.com/qwertyforce/places365_tagger_web)

## Installation  
1. Clone the repository  
2. ```npm install```
## Usage
```npm run build```  -> builds .ts files  
```npm run start``` -> starts the server  
## Docker  
```docker network create --driver bridge ambience_net ``` -> create network for microservices  
```docker build -t qwertyforce/ambience:1.0.0 --network host -t qwertyforce/ambience:latest ./``` -> build  
```docker run -d --rm -p 127.0.0.1:44444:44444 --network=ambience_net --name ambience qwertyforce/ambience:1.0.0``` -> run as daemon

