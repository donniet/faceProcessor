package main

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"image"
	"image/jpeg"
	"log"
	"os"
	"time"

	"github.com/donniet/raspividWrapper/videoService"
	"github.com/golang/protobuf/ptypes/empty"

	tf_core_framework "tensorflow/core/framework"
	pb "tensorflow_serving/apis"

	google_protobuf "github.com/golang/protobuf/ptypes/wrappers"

	"google.golang.org/grpc"
)

var (
	throttle                = 1000 * time.Millisecond
	faceDetectionAddr       = "localhost:8500"
	modelName               = "face_detection"
	signatureName           = "serving_default"
	modelVersion      int64 = 1
	inputName               = "image_tensor"
	boxesOutput             = "detection_boxes"
	scoresOutput            = "detection_scores"
	numOutput               = "num_detections"
	framesGRPCAddress       = "mirror.local:5555"
	width                   = 1640
	height                  = 1232
)

func init() {
	flag.DurationVar(&throttle, "throttle", throttle, "throttle motion detection to save CPU")
	flag.StringVar(&faceDetectionAddr, "servinggrpc", faceDetectionAddr, "GRPC address to TensorFlow Serving for Face Processing")
	flag.StringVar(&modelName, "model", modelName, "TensorFlow Serving Model Name")
	flag.StringVar(&signatureName, "signature", signatureName, "TensorFlow Serving Model Signature Name")
	flag.Int64Var(&modelVersion, "modelversion", modelVersion, "TensorFlow Serving Model Version")
	flag.StringVar(&inputName, "modelinput", inputName, "TensorFlow Serving Model Input Name")
	flag.StringVar(&boxesOutput, "boxesoutput", boxesOutput, "TensorFlow Serving Model Output Boxes Name")
	flag.StringVar(&scoresOutput, "scoresoutput", scoresOutput, "TensorFlow Serving Model Output Scores Name")
	flag.StringVar(&numOutput, "numoutput", numOutput, "TensorFlow Serving Model Output Num Name")
	flag.StringVar(&framesGRPCAddress, "framesgrpc", framesGRPCAddress, "GRPC address of frames")
	flag.StringVar(&modelName, "modelName", modelName, "name of motion processing model in tensorflow serving")
}

func main() {
	flag.Parse()

	conn, err := grpc.Dial(faceDetectionAddr, grpc.WithInsecure())
	if err != nil {
		log.Fatal(err)
	}
	defer conn.Close()

	faceClient := pb.NewPredictionServiceClient(conn)

	frameConn, err := grpc.Dial(framesGRPCAddress,
		grpc.WithInsecure(),
		grpc.WithDefaultCallOptions(grpc.MaxSendMsgSizeCallOption{0x800000}))
	if err != nil {
		log.Fatal(err)
	}
	defer frameConn.Close()

	framesClient := videoService.NewVideoClient(frameConn)

	imgCount := 0

	for {
		frame, err := framesClient.FrameJPEG(context.Background(), &empty.Empty{})
		if err != nil {
			log.Printf("error receiving frame: %v", err)
			break
		}

		img, err := jpeg.Decode(bytes.NewReader(frame.GetData()))
		if err != nil {
			log.Printf("error decoding jpeg: %v", err)
			break
		}

		rgb := FromImage(img)

		req := &pb.PredictRequest{
			ModelSpec: &pb.ModelSpec{
				Name:          modelName,
				SignatureName: signatureName,
				Version: &google_protobuf.Int64Value{
					Value: modelVersion,
				},
			},
			Inputs: map[string]*tf_core_framework.TensorProto{
				inputName: &tf_core_framework.TensorProto{
					Dtype: tf_core_framework.DataType_DT_UINT8,
					TensorShape: &tf_core_framework.TensorShapeProto{
						Dim: []*tf_core_framework.TensorShapeProto_Dim{
							&tf_core_framework.TensorShapeProto_Dim{Size: int64(1)},
							&tf_core_framework.TensorShapeProto_Dim{Size: int64(width)},
							&tf_core_framework.TensorShapeProto_Dim{Size: int64(height)},
							&tf_core_framework.TensorShapeProto_Dim{Size: int64(3)},
						},
					},
					TensorContent: rgb.Pix,
				},
			},
		}

		res, err := faceClient.Predict(context.Background(), req)
		if err != nil {
			log.Printf("error from face detector: %v", err)
			break
		}

		// log.Printf("num detections: %f", res.Outputs[numOutput].GetFloatVal()[0])
		log.Printf("first confidence: %f", res.Outputs[scoresOutput].GetFloatVal()[0])

		scores := res.Outputs[scoresOutput].GetFloatVal()
		categories := res.Outputs["detection_classes"].GetFloatVal()
		rects := res.Outputs[boxesOutput].GetFloatVal()

		for i := 0; i < 100; i++ {
			// log.Printf("score of %d: %f", i, scores[i])
			// log.Printf("categories of %d: %f", i, categories[i])

			if categories[i] > 1.5 {
				continue
			}

			if scores[i] < 0.6 {
				continue
			}

			rect := rects[i*4 : i*4+4]

			log.Printf("rect: %f %f %f %f", rect[0], rect[1], rect[2], rect[3])
			x0 := int(rect[0] * float32(rgb.Bounds().Dx()))
			y0 := int(rect[1] * float32(rgb.Bounds().Dy()))
			x1 := x0 + int(rect[2]*float32(rgb.Bounds().Dx()))
			y1 := y0 + int(rect[2]*float32(rgb.Bounds().Dy()))
			face := rgb.SubImage(image.Rect(x0, y0, x1, y1))

			f, err := os.OpenFile(fmt.Sprintf("image%05d.jpg", imgCount), os.O_CREATE|os.O_WRONLY, 0660)
			if err != nil {
				log.Printf("error opening file: %v", err)
			} else if jpeg.Encode(f, face, nil); err != nil {
				log.Printf("error writing out jpeg of face: %v", err)
			} else {
				imgCount++
			}
		}
	}

	log.Printf("closing...")
}
