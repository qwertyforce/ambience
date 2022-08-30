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

async function calculate_image_text_features(image_id: number, image: Buffer) {
    const form = new FormData()
    form.append('image', image, { filename: 'document' }) //hack to make nodejs buffer work with form-data
    form.append('image_id', image_id.toString())
    const status = await axios.post(`${config.image_text_features_microservice_url}/calculate_image_text_features`, form.getBuffer(), {
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

// async function calculate_text_features(image_id: number, image: Buffer) {
//     const form = new FormData()
//     form.append('image', image, { filename: 'document' }) //hack to make nodejs buffer work with form-data
//     form.append('image_id', image_id.toString())
//     const status = await axios.post(`${config.text_microservice_url}/calculate_text_features`, form.getBuffer(), {
//         maxContentLength: Infinity,
//         maxBodyLength: Infinity,
//         headers: {
//             ...form.getHeaders()
//         }
//     })
//     return status.data
// }

interface ImageSearchProps {
    image: Buffer,
    k?: number,
    distance_threshold?: number
    k_clusters?: number
    knn_min_matches?: number
    matching_threshold?: number
    aqe_n?: number
    aqe_alpha?: number
    use_smnn_matching?: number
    smnn_match_threshold?: number
    use_ransac?: number
}

async function phash_get_similar_images_by_image_buffer({ image, k, distance_threshold }: ImageSearchProps) {
    try {
        const form = new FormData()
        if (k) {
            form.append('k', k.toString())
        } else if (distance_threshold) {
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

async function global_features_get_similar_images_by_image_buffer({ image, k, distance_threshold, aqe_n, aqe_alpha }: ImageSearchProps) {
    try {
        const form = new FormData()
        if (k) {
            form.append('k', k.toString())
        } else if (distance_threshold) {
            form.append('distance_threshold', distance_threshold.toString())
        }
        if (aqe_n) {
            form.append('aqe_n', aqe_n.toString())
        }
        if (aqe_alpha) {
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

async function image_text_features_get_similar_images_by_image_buffer({ image, k, distance_threshold, aqe_n, aqe_alpha }: ImageSearchProps) {
    try {
        const form = new FormData()
        if (k) {
            form.append('k', k.toString())
        } else if (distance_threshold) {
            form.append('distance_threshold', distance_threshold.toString())
        }
        if (aqe_n) {
            form.append('aqe_n', aqe_n.toString())
        }
        if (aqe_alpha) {
            form.append('aqe_alpha', aqe_alpha.toString())
        }
        form.append('image', image, { filename: 'document' }) //hack to make nodejs buffer work with form-data
        const res = await axios.post(`${config.image_text_features_microservice_url}/image_text_features_get_similar_images_by_image_buffer`, form.getBuffer(), {
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

async function color_get_similar_images_by_image_buffer({ image, k, distance_threshold }: ImageSearchProps) {
    try {
        const form = new FormData()
        if (k) {
            form.append('k', k.toString())
        } else if (distance_threshold) {
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

async function local_features_get_similar_images_by_image_buffer({ image,
    k, k_clusters, knn_min_matches, matching_threshold, use_smnn_matching, smnn_match_threshold, use_ransac }: ImageSearchProps) {
    try {
        const form = new FormData()
        if (k) {
            form.append('k', k.toString())
        }
        if (k_clusters) {
            form.append('k_clusters', k_clusters.toString())
        }
        if (knn_min_matches) {
            form.append('knn_min_matches', knn_min_matches.toString())
        }
        if (matching_threshold) {
            form.append('matching_threshold', matching_threshold.toString())
        }
        if (use_smnn_matching) {
            form.append('use_smnn_matching', use_smnn_matching.toString())
        }
        if (smnn_match_threshold) {
            form.append('smnn_match_threshold', smnn_match_threshold.toString())
        }
        if (use_ransac) {
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

// async function text_get_similar_images_by_image_buffer({image,k,distance_threshold}:ImageSearchProps) {
//     try {
//         const form = new FormData()
//         if(k){
//             form.append('k', k.toString())
//         }else if(distance_threshold){
//             form.append('distance_threshold', distance_threshold.toString())
//         }

//         form.append('image', image, { filename: 'document' }) //hack to make nodejs buffer work with form-data
//         const res = await axios.post(`${config.text_microservice_url}/text_get_similar_images_by_image_buffer`, form.getBuffer(), {
//             maxContentLength: Infinity,
//             maxBodyLength: Infinity,
//             headers: {
//                 ...form.getHeaders()
//             }
//         })
//         return res.data
//     } catch (err) {
//         console.log(err)
//         return []
//     }
// }

async function get_similar_images(image: Buffer,find_duplicate:boolean) {
    let phash_res:any = []
    let global_features_res:any = []
    let local_features_res:any = [] 
    let color_res:any = [] 
    let image_text_res:any  = [];
    // const text_res = await text_get_similar_images_by_image_buffer({image:image,k:200})
    if(find_duplicate){
        local_features_res = (await local_features_get_similar_images_by_image_buffer({
            image: image, k: 5, k_clusters: 15,
            knn_min_matches: 4, matching_threshold: 0.8,
            use_smnn_matching: 1, use_ransac: 1
        })).filter((el:any)=>el["matches"]>=8)
    }else{
        [phash_res, global_features_res,local_features_res,color_res,image_text_res]  = (await Promise.allSettled([
            phash_get_similar_images_by_image_buffer({ image: image, k: 200 }),
            global_features_get_similar_images_by_image_buffer({ image: image, k: 200 }),
            local_features_get_similar_images_by_image_buffer({
                image: image, k: 200, k_clusters: 10,
                knn_min_matches: 4, matching_threshold: 0.8,
                use_smnn_matching: 1, use_ransac: 1
            }),
            color_get_similar_images_by_image_buffer({ image: image, k: 200 }),
            image_text_features_get_similar_images_by_image_buffer({ image: image, k: 200 })
        ])).map((promise: any) => promise.value)
    }
    console.log("==================")
    console.log("local_features")
    console.log(local_features_res)
    console.log("==================")

    console.log("==================")
    console.log("phash")
    console.log(phash_res)
    console.log("==================")

    console.log("==================")
    console.log("global_features")
    console.log(global_features_res)
    console.log("==================")

    console.log("==================")
    console.log("color")
    console.log(color_res)
    console.log("==================")

    // console.log("==================")
    // console.log("text")
    // console.log(text_res)
    // console.log("==================")

    const _unified_res: { [key: string]: any } = {}
    const unified_res = []
    for (const data_source of [[phash_res, "phash_res"],
    [global_features_res, "global_features_res"], [local_features_res, "local_features_res"],
    [color_res, "color_res"], [image_text_res, "image_text_res"]]) {
        for (let i = 0; i < data_source[0].length; i++) {
            const res = data_source[0][i]
            if (!_unified_res[res.image_id]) {
                _unified_res[res.image_id] = {}
            }
            _unified_res[res.image_id][data_source[1]] = i
        }
    }
    console.log(_unified_res)
    for (const image of Object.entries(_unified_res)) {
        if (Object.entries(image[1]).length > 1) {
            let score = 0
            for (const data_source in image[1]) {
                const idx_in_results = image[1][data_source] + 1
                let multiplier = 1
                switch (data_source) {
                    case "global_features_res":
                        multiplier = 1
                        break
                    case "local_features_res":
                        multiplier = 1
                        break
                    case "color_res":
                        multiplier = 1
                        break
                    case "image_text_res":
                        multiplier = 1
                        break
                    case "phash_res":
                        multiplier = 1
                        break
                }
                score += multiplier * (1 / idx_in_results)
            }
            unified_res.push({ image_id: parseInt(image[0]), score: score })
        }
    }
    const all_results: any = {}
    for (const data_source of [[phash_res, "phash_res"],
    [global_features_res, "global_features_res"], [local_features_res, "local_features_res"],
    [color_res, "color_res"], [image_text_res, "image_text_res"]]) {
        if (data_source[0].length!==0){
            all_results[data_source[1]] = data_source[0].slice(0,20)
        }
    }
    if (unified_res.length !== 0) {
        all_results.unified_res = unified_res.sort((a:any,b:any)=>-(a.score-b.score))
    }
    return all_results
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

async function delete_image_text_features_by_id(image_id: number) {
    const status = await axios.post(`${config.image_text_features_microservice_url}/delete_image_text_features`, { image_id: image_id })
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

// async function delete_text_features_by_id(image_id: number) {
//     const status = await axios.post(`${config.text_microservice_url}/delete_text_features`, { image_id: image_id })
//     return status.data
// }
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
async function calculate_all_image_features(image_id: number, image_buffer: Buffer) {
    return Promise.allSettled([
        calculate_global_features(image_id, image_buffer),
        calculate_local_features(image_id, image_buffer),
        calculate_color_features(image_id, image_buffer),
        calculate_phash_features(image_id, image_buffer),
        calculate_image_text_features(image_id, image_buffer),
        // calculate_text_features(image_id, image_buffer)
    ])
}

async function delete_all_image_features(image_id: number) {
    return Promise.allSettled([
        delete_global_features_by_id(image_id),
        delete_local_features_by_id(image_id),
        delete_color_features_by_id(image_id),
        delete_phash_features_by_id(image_id),
        delete_image_text_features_by_id(image_id)
        // delete_text_features_by_id(image_id)
    ])
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
export default { calculate_all_image_features, delete_all_image_features, get_similar_images }