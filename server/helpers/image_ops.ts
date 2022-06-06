import config from "../../config/config"
import FormData from 'form-data'
import axios from 'axios'


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
async function calculate_phash_features(image_id: number, image: Buffer) {
    const form = new FormData()
    form.append('image', image, { filename: 'document' }) //hack to make nodejs buffer work with form-data
    form.append('image_id', image_id.toString())
    const status = await axios.post(`${config.phash_microservice_url}/calculate_phash_features`, form.getBuffer(), {
        maxContentLength: Infinity,
        maxBodyLength: Infinity,
        headers: {
            ...form.getHeaders()
        }
    })
    return status.data
}

async function calculate_local_features(image_id: number, image: Buffer) {
    const form = new FormData()
    form.append('image', image, { filename: 'document' }) //hack to make nodejs buffer work with form-data
    form.append('image_id', image_id.toString())
    const status = await axios.post(`${config.local_features_microservice_url}/calculate_local_features`, form.getBuffer(), {
        maxContentLength: Infinity,
        maxBodyLength: Infinity,
        headers: {
            ...form.getHeaders()
        }
    })
    return status.data
}
async function calculate_global_features(image_id: number, image: Buffer) {
    const form = new FormData()
    form.append('image', image, { filename: 'document' }) //hack to make nodejs buffer work with form-data
    form.append('image_id', image_id.toString())
    const status = await axios.post(`${config.global_features_microservice_url}/calculate_global_features`, form.getBuffer(), {
        maxContentLength: Infinity,
        maxBodyLength: Infinity,
        headers: {
            ...form.getHeaders()
        }
    })
    return status.data
}

async function calculate_color_features(image_id: number, image: Buffer) {
    const form = new FormData()
    form.append('image', image, { filename: 'document' }) //hack to make nodejs buffer work with form-data
    form.append('image_id', image_id.toString())
    const status = await axios.post(`${config.color_microservice_url}/calculate_color_features`, form.getBuffer(), {
        maxContentLength: Infinity,
        maxBodyLength: Infinity,
        headers: {
            ...form.getHeaders()
        }
    })
    return status.data
}

interface ImageSearchProps{
    image:Buffer,
    k?:number,
    distance_threshold?:number
    k_clusters?:number
    min_matches?:number
    matching_threshold?:number
    aqe_n?:number
    aqe_alpha?:number
    use_snn_matching?:number
    snn_match_threshold?:number
    use_ransac?:number
}

async function phash_get_similar_images_by_image_buffer({image,k,distance_threshold}:ImageSearchProps) {
    try {
        const form = new FormData()
        if(k){
            form.append('k', k.toString())
        }else if(distance_threshold){
            form.append('distance_threshold', distance_threshold.toString())
        }
        
        form.append('image', image, { filename: 'document' }) //hack to make nodejs buffer work with form-data
        const res = await axios.post(`${config.phash_microservice_url}/phash_get_similar_images_by_image_buffer`, form.getBuffer(), {
            maxContentLength: Infinity,
            maxBodyLength: Infinity,
            headers: {
                ...form.getHeaders()
            }
        })
        return res.data
    } catch (err) {
        console.log(err)
        return []
    }
}

async function global_features_get_similar_images_by_image_buffer({image,k,distance_threshold,aqe_n,aqe_alpha}:ImageSearchProps) {
    try {
        const form = new FormData()
        if(k){
            form.append('k', k.toString())
        }else if(distance_threshold){
            form.append('distance_threshold', distance_threshold.toString())
        }
        if(aqe_n){
            form.append('aqe_n', aqe_n.toString())
        }
        if(aqe_alpha){
            form.append('aqe_alpha', aqe_alpha.toString())
        }
        form.append('image', image, { filename: 'document' }) //hack to make nodejs buffer work with form-data
        const res = await axios.post(`${config.global_features_microservice_url}/global_features_get_similar_images_by_image_buffer`, form.getBuffer(), {
            maxContentLength: Infinity,
            maxBodyLength: Infinity,
            headers: {
                ...form.getHeaders()
            }
        })
        return res.data
    } catch (err) {
        console.log(err)
        return []
    }
}

async function color_get_similar_images_by_image_buffer({image,k,distance_threshold}:ImageSearchProps) {
    try {
        const form = new FormData()
        if(k){
            form.append('k', k.toString())
        }else if(distance_threshold){
            form.append('distance_threshold', distance_threshold.toString())
        }
        
        form.append('image', image, { filename: 'document' }) //hack to make nodejs buffer work with form-data
        const res = await axios.post(`${config.color_microservice_url}/color_get_similar_images_by_image_buffer`, form.getBuffer(), {
            maxContentLength: Infinity,
            maxBodyLength: Infinity,
            headers: {
                ...form.getHeaders()
            }
        })
        return res.data
    } catch (err) {
        console.log(err)
        return []
    }
}

async function local_features_get_similar_images_by_image_buffer({image,
    k,k_clusters,min_matches,matching_threshold,use_snn_matching,snn_match_threshold,use_ransac}:ImageSearchProps) {
    try {
        const form = new FormData()
        if(k){
            form.append('k', k.toString())
        }
        if(k_clusters){
            form.append('k_clusters', k_clusters.toString())
        }
        if(min_matches){
            form.append('min_matches', min_matches.toString())
        }
        if(matching_threshold){
            form.append('matching_threshold', matching_threshold.toString())
        }
        if(use_snn_matching){
            form.append('use_snn_matching', use_snn_matching.toString())
        }
        if(snn_match_threshold){
            form.append('snn_match_threshold', snn_match_threshold.toString())
        }
        if(use_ransac){
            form.append('use_ransac', use_ransac.toString())
        }
        
        form.append('image', image, { filename: 'document' }) //hack to make nodejs buffer work with form-data
        const res = await axios.post(`${config.local_features_microservice_url}/local_features_get_similar_images_by_image_buffer`, form.getBuffer(), {
            maxContentLength: Infinity,
            maxBodyLength: Infinity,
            headers: {
                ...form.getHeaders()
            }
        })
        return res.data
    } catch (err) {
        console.log(err)
        return []
    }
}

async function text_get_similar_images_by_image_buffer({image,k,distance_threshold}:ImageSearchProps) {
    try {
        const form = new FormData()
        if(k){
            form.append('k', k.toString())
        }else if(distance_threshold){
            form.append('distance_threshold', distance_threshold.toString())
        }
        
        form.append('image', image, { filename: 'document' }) //hack to make nodejs buffer work with form-data
        const res = await axios.post(`${config.text_microservice_url}/text_get_similar_images_by_image_buffer`, form.getBuffer(), {
            maxContentLength: Infinity,
            maxBodyLength: Infinity,
            headers: {
                ...form.getHeaders()
            }
        })
        return res.data
    } catch (err) {
        console.log(err)
        return []
    }
}

async function get_similar_images(image: Buffer) {
    const phash_res = await phash_get_similar_images_by_image_buffer({image:image,k:200})
    const global_features_res = await global_features_get_similar_images_by_image_buffer({image:image,k:200})
    const local_features_res = await local_features_get_similar_images_by_image_buffer({
        image:image,k:200, k_clusters:10,
        min_matches:4, matching_threshold:1.1,
        use_snn_matching:1, use_ransac:1})
    const color_res = await color_get_similar_images_by_image_buffer({image:image,k:200})
    const text_res = await text_get_similar_images_by_image_buffer({image:image,k:200})

    console.log("==================")
    console.log("phash")
    console.log(local_features_res)
    console.log("==================")

    console.log("==================")
    console.log("phash")
    console.log(phash_res)
    console.log("==================")

    console.log("==================")
    console.log("nn")
    console.log(global_features_res)
    console.log("==================")

    console.log("==================")
    console.log("color")
    console.log(color_res)
    console.log("==================")

    console.log("==================")
    console.log("text")
    console.log(text_res)
    console.log("==================")
    return {phash:phash_res,global_features:global_features_res,local_features:local_features_res,color:color_res,text:text_res}
} 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////////
async function delete_local_features_by_id(image_id: number) {
    const status = await axios.post(`${config.local_features_microservice_url}/delete_local_features`, { image_id: image_id })
    return status.data
}

async function delete_global_features_by_id(image_id: number) {
    const status = await axios.post(`${config.global_features_microservice_url}/delete_global_features`, { image_id: image_id })
    return status.data
}

async function delete_color_features_by_id(image_id: number) {
    const status = await axios.post(`${config.color_microservice_url}/delete_color_features`, { image_id: image_id })
    return status.data
}

async function delete_phash_features_by_id(image_id: number) {
    const status = await axios.post(`${config.phash_microservice_url}/delete_phash_features`, { image_id: image_id })
    return status.data
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
async function calculate_all_image_features(image_id: number, image_buffer: Buffer) {
    return Promise.allSettled([
        calculate_global_features(image_id, image_buffer),
        calculate_local_features(image_id, image_buffer),
        calculate_color_features(image_id, image_buffer),
        calculate_phash_features(image_id, image_buffer),
    ])
}

async function delete_all_image_features(image_id: number) {
    return Promise.allSettled([
        delete_global_features_by_id(image_id),
        delete_local_features_by_id(image_id),
        delete_color_features_by_id(image_id),
        delete_phash_features_by_id(image_id)
    ])
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
export default { calculate_all_image_features, delete_all_image_features, get_similar_images}