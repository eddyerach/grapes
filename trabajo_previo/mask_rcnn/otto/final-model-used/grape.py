from flask import Flask, request, jsonify
from urllib.parse import urlparse
import threading, queue
import random
import time
import requests
import cv2
import sys

import boto3
import psycopg2
from psycopg2 import sql

from PIL import Image
from io import BytesIO
import numpy as np

import ml_rcnn

import os, uuid
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

from time import perf_counter


app = Flask(__name__)

API_KEY = "12345"
AZURE_STORAGE_CONNECTION_STRING ="DefaultEndpointsProtocol=https;AccountName=grapesblob;AccountKey=AuPGit9tvo/+qyK2J7cUgYYYTRgi/hlH6jKfxyJ7LfM/qdeC39D8lG6DzxY2vcVBTa6SLpKBUVcckfWrsmL9Ow==;EndpointSuffix=core.windows.net"


#========================= Azure Storage =========================


def set_connection_str(connect_str):

    global  local_path, blob_service_client
    local_path = '.'   # local_path to download and upload files to azure storage

    try:
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        
    except Exception as ex:
        print('Exception:')
        print(ex)


def download_file_from_container(file_name, container_name = 'images'):

    try:
        download_file_path = os.path.join(local_path, file_name)
        blob_client = blob_service_client.get_blob_client(container = container_name, blob = file_name)

        with open(download_file_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())

        image = Image.open(download_file_path)
        return np.array(image)

    except Exception as ex:
        print('Exception:')
        print(ex)
        return []


def upload_file_to_container(file_array, local_file, container_name = 'processed-images'):
        
    try:
        image = Image.fromarray(file_array)
        file_stream = BytesIO()
        image.save(file_stream, format='jpeg')

        upload_file_path = os.path.join(local_path, local_file)
        blob_client = blob_service_client.get_blob_client(container = container_name, blob = local_file)

        with open(upload_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

    except Exception as ex:
        print('Exception:')
        print(ex)


#========================= S3 AWS Storage =========================


def uri_validator(x):
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc])
    except:
        return False


def read_image_from_s3(bucket, key, region_name='us-east-1'):
    """Load image file from s3.

    Parameters
    ----------
    bucket: string
        Bucket name
    key : string
        Path in s3

    Returns
    -------
    np array
        Image array
    """
    s3 = boto3.resource('s3', region_name)
    bucket = s3.Bucket(bucket)
    try:
        object = bucket.Object(key)
        response = object.get()
        file_stream = response['Body']
        im = Image.open(file_stream)
        return np.array(im)
    except BaseException as error:
        return []
    

def write_image_to_s3(img_array, bucket, key, region_name='us-east-1'):
    """Write an image array into S3 bucket

    Parameters
    ----------
    bucket: string
        Bucket name
    key : string
        Path in s3

    Returns
    -------
    None
    """
    s3 = boto3.resource('s3', region_name)
    bucket = s3.Bucket(bucket)
    object = bucket.Object(key)
    file_stream = BytesIO()
    im = Image.fromarray(img_array)
    im.save(file_stream, format='jpeg')
    object.put(Body=file_stream.getvalue())


#========================= Work Thread =========================

work_queue = queue.Queue()
STOP_WORK = False

set_connection_str(AZURE_STORAGE_CONNECTION_STRING)

#forma {img_id: {timestamp: {count: int, detection_url: str}, timestamp2: ..}, img_id2: ... }
#result_database = {}
#res_db_lock = threading.Lock()


def add_to_queue(img_url,img_id,timestamp):
    #asume que parametros ya fueron validados
    global work_queue
    work_queue.put({"img_id":img_id,"img_url":img_url,"timestamp":timestamp})


def notify_thread(data):
    #print("DATA: ",data)
    try:
        re = requests.post("http://127.0.0.1:6000/api/ready",json=data)
        return
    except BaseException as error:
        print('An exception occurred: {}'.format(error))
        return


def work_thread():
    global res_db_lock
    #global result_database
    global STOP_WORK
    global work_queue

    ml_path = "/home/azureuser/grape/"
    model = ml_rcnn.MLModel(ml_path)

    add_query = "INSERT INTO detections (img_id, img_url, last_detected, last_status, grape_count,detect_url) VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT(img_id) DO UPDATE SET img_url = EXCLUDED.img_url,last_detected = EXCLUDED.last_detected,last_status = EXCLUDED.last_status,grape_count = EXCLUDED.grape_count,detect_url = EXCLUDED.detect_url;"

    while(not STOP_WORK):
        img_data = work_queue.get()
        #asumamos que img_data siempre viene bien (por ahora)
        #im = read_image_from_s3("uvas1",img_data["img_url"]) #cv2.imread(img_data["img_url"]) # DESCOMENTAR para usar S3
        im = download_file_from_container(str(img_data["img_url"]))

        #not found
        if(im == []):
            result = {"status":20,"count":0,"detection_url":""}
        else:
            output = model.predict(im)
            detect_url = str(img_data["img_id"])+".jpg"#ml_path+"output/"+str(img_data["img_id"])+".jpg"#+"_"+str(img_data["timestamp"])+".jpg"#"https://www.imagenesdeuvas.com/"+str(img_data["img_id"])+".jpg"
            cv2.imwrite(detect_url,output["detect"])
            result = {"status":0,"count":output["count"],"detection_url":detect_url}
            #write_image_to_s3(output["detect"],"uvasdetect1",str(img_data["img_id"])+".jpg")  # DESCOMENTAR para usar S3
            upload_file_to_container(output["detect"], str(img_data["img_url"]))
            
            if os.path.exists(str(img_data["img_url"])):
                os.remove(str(img_data["img_url"]))
        
        '''
        res_db_lock.acquire()
        if img_data["img_id"] in result_database:#ya existe la img en la db
            result_database[img_data["img_id"]][img_data["timestamp"]] = result
        else:
            result_database[img_data["img_id"]] = {img_data["timestamp"]: result} 

        print(result_database)
        res_db_lock.release()
        '''

        db_conn = None
        try:
            db_conn = psycopg2.connect(user="ubuntu",
                                       password="uvas123",
                                       host="127.0.0.1",
                                       port="5432",
                                       database="grape")
            cursor = db_conn.cursor()
            cursor.execute(add_query,(img_data["img_id"],
                                      img_data["img_url"],
                                      img_data["timestamp"],
                                      result["status"],
                                      result["count"],
                                      result["detection_url"]))
            db_conn.commit()
            cursor.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error, file = sys.stderr)
        finally:
            if db_conn is not None:
                db_conn.close()
        

        #notificamos el resultado
        notify = dict(img_data,**result) #hack
        notify_th = threading.Thread(target=notify_thread,args=[notify])
        notify_th.daemon = True
        notify_th.start()


def fetch_result(img_id):
    global res_db_lock
    fetch_query = "SELECT * FROM detections WHERE img_id = %s;"

    #query a la db
    db_conn = None
    results = {}

    try:
        db_conn = psycopg2.connect(user="ubuntu",
                                   password="uvas123",
                                   host="127.0.0.1",
                                   port="5432",
                                   database="grape")
        cursor = db_conn.cursor()
        cursor.execute(fetch_query,(img_id,))
        query_res = cursor.fetchone()
        results = {"status": query_res[3],"count":query_res[4],"detection_url":query_res[5]}
        cursor.close()
        return results

    except (Exception, psycopg2.DatabaseError) as error:
        print(error, file = sys.stderr)
        return {}

    finally:
        if db_conn is not None:
            db_conn.close()
        return results

    '''
    res_db_lock.acquire()
    if img_id not in result_database:
        res_db_lock.release()
        return {}

    res_dict = result_database[img_id].copy()
    res_db_lock.release()

    return res_dict[max(res_dict)]
    '''



work_th = threading.Thread(target=work_thread)
work_th.daemon = True
work_th.start()



##############################################################



@app.route("/api/add",methods=['POST'])
def API_add():
    data = request.get_json(force=True)

    #wrong structure
    if "api_key" not in data or "img_id" not in data or "img_url" not in data:
        return "",400

    #wrong types
    if type(data["api_key"]) is not str or type(data["img_id"]) is not int or type(data["img_url"]) is not str:
        return "",400

    #malformed url
    #if not uri_validator(data["img_url"]):
    #    return "",400

    #wrong key
    if data["api_key"] != API_KEY:
        return "",400


    timestmp = int(time.time())
    add_to_queue(data["img_url"],data["img_id"],timestmp)

 
    return jsonify(timestamp=timestmp)



@app.route("/api/fetch",methods=['POST'])
def API_fetch():
    data = request.get_json(force=True)

    #wrong structure
    if "api_key" not in data or "img_id" not in data:
        return "",400

    #wrong types
    if type(data["api_key"]) is not str or type(data["img_id"]) is not int:
        return "",400

    #wrong key
    if data["api_key"] != API_KEY:
        return "",400

    
    detect_res = fetch_result(data["img_id"])

    if detect_res == {}:
        return jsonify(status=10)


    return jsonify(detect_res)



