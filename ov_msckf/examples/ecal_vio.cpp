#include <csignal>
#include <memory>
#include <iostream>
#include <atomic>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "ros/ROS2Visualizer.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <ecal/ecal.h>
#include <capnp/serialize.h>
#include <kj/array.h>
#include "imu.capnp.h"
#include "image.capnp.h"
#include "header.capnp.h"

using namespace ov_msckf;

// Global variables
std::shared_ptr<VioManager> sys;
std::shared_ptr<ROS2Visualizer> viz;
std::shared_ptr<rclcpp::Node> ros_node;
std::atomic<bool> shutdown_flag(false);

// Configuration struct for eCAL topics
struct EcalConfig {
  std::string imu_topic = "/imu0";
  std::vector<std::string> camera_topics;
};

class EcalVioNode {
public:
  EcalVioNode(const EcalConfig& config, std::shared_ptr<VioManager> vio_manager)
    : config_(config), sys_(vio_manager) {}

  bool initialize() {
    // Create IMU subscriber
    imu_subscriber_ = std::make_unique<eCAL::CSubscriber>(config_.imu_topic);
    if (!imu_subscriber_->IsCreated()) {
      PRINT_ERROR(RED "Failed to create IMU subscriber for topic: %s\n" RESET, config_.imu_topic.c_str());
      return false;
    }

    imu_subscriber_->AddReceiveCallback(std::bind(&EcalVioNode::onImuMessage, this, std::placeholders::_1, std::placeholders::_2));

    // Initialize vectors for multiple cameras
    camera_subscribers_.resize(config_.camera_topics.size());
    buffered_images_.resize(config_.camera_topics.size());
    buffered_timestamps_.resize(config_.camera_topics.size(), -1.0);
    image_counts_.resize(config_.camera_topics.size(), 0);

    // Create camera subscribers
    for (size_t i = 0; i < config_.camera_topics.size(); ++i) {
      const std::string& topic = config_.camera_topics[i];
      camera_subscribers_[i] = std::make_unique<eCAL::CSubscriber>(topic);
      if (!camera_subscribers_[i]->IsCreated()) {
        PRINT_ERROR(RED "Failed to create camera subscriber for topic: %s\n" RESET, topic.c_str());
        return false;
      }

      camera_subscribers_[i]->AddReceiveCallback(
        std::bind(&EcalVioNode::onImageMessage, this, std::placeholders::_1, std::placeholders::_2, i)
      );
    }

    return true;
  }

  void shutdown() {
    shutdown_flag.store(true);
  }

private:
  void onImuMessage(const char* topic_name, const struct eCAL::SReceiveCallbackData* data);
  void onImageMessage(const char* topic_name, const struct eCAL::SReceiveCallbackData* data, size_t cam_idx);
  ov_core::ImuData convertImuMessage(const vkc::Imu::Reader& imu_msg);
  cv::Mat convertImageMessage(const vkc::Image::Reader& image_msg);
  sensor_msgs::msg::Image::SharedPtr convertToRosImage(const vkc::Image::Reader& image_msg, const std::string& frame_id);
  void processCameraImages();
  bool allCamerasSynchronized(double sync_threshold = 0.01);

  EcalConfig config_;
  std::shared_ptr<VioManager> sys_;

  std::unique_ptr<eCAL::CSubscriber> imu_subscriber_;
  std::vector<std::unique_ptr<eCAL::CSubscriber>> camera_subscribers_;

  std::vector<cv::Mat> buffered_images_;
  std::vector<double> buffered_timestamps_;

  size_t imu_message_count_ = 0;
  std::vector<size_t> image_counts_;
};

void signal_callback_handler(int signum) {
    shutdown_flag.store(true);
    if (viz) {
        viz->visualize_final();
    }
    eCAL::Finalize();
    if (rclcpp::ok()) {
        rclcpp::shutdown();
    }
    std::exit(signum);
}

int main(int argc, char** argv)
{
  // Ensure we have a path, if the user passes it then we should use it
  std::string config_path = "unset_path_to_config.yaml";
  if (argc > 1) {
    config_path = argv[1];
  }

  // Initialize ROS2
  rclcpp::init(argc, argv);
  ros_node = rclcpp::Node::make_shared("ecal_vio_visualizer");

  // Initialize eCAL
  eCAL::Initialize(argc, argv, "ecal_vio");

  // Set up signal handler
  signal(SIGINT, signal_callback_handler);

  // Load OpenVINS configuration
  std::shared_ptr<ov_core::YamlParser> parser;
  try {
    parser = std::make_shared<ov_core::YamlParser>(config_path);
    parser->set_node(ros_node);
  } catch(const std::exception& e) {
    PRINT_ERROR(RED "Failed to load configuration: %s\n" RESET, e.what());
    eCAL::Finalize();
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }

  VioManagerOptions params;
  params.print_and_load(parser);
  params.num_opencv_threads = 0; // for repeatability
  params.use_multi_threading_pubs = false;
  params.use_multi_threading_subs = false;
  sys = std::make_shared<VioManager>(params);

  // Ensure we read in all parameters required
  if (!parser->successful()) {
    PRINT_ERROR(RED "unable to parse all parameters, please fix\n" RESET);
    eCAL::Finalize();
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }

  // Configure eCAL topics
  EcalConfig ecal_config;

  // Get IMU topic from config
  parser->parse_external("relative_config_imu", "imu0", "rostopic", ecal_config.imu_topic);

  // Get camera topics from config
  ecal_config.camera_topics.resize(params.state_options.num_cameras);
  for (int i = 0; i < params.state_options.num_cameras; i++) {
    std::string cam_topic;
    parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "rostopic", cam_topic);
    ecal_config.camera_topics[i] = cam_topic;
  }

  // Create ROS2 visualizer
  viz = std::make_shared<ROS2Visualizer>(ros_node, sys);

  // Create eCAL VIO node
  auto ecal_node = std::make_unique<EcalVioNode>(ecal_config, sys);

  if (!ecal_node->initialize()) {
    PRINT_ERROR(RED "Failed to initialize eCAL VIO node\n" RESET);
    eCAL::Finalize();
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }

  PRINT_INFO("eCAL VIO node initialized successfully\n");
  PRINT_INFO("ROS2 visualizer initialized successfully\n");
  PRINT_INFO("Listening for IMU data on: %s\n", ecal_config.imu_topic.c_str());
  for (size_t i = 0; i < ecal_config.camera_topics.size(); ++i) {
    PRINT_INFO("Listening for camera %zu data on: %s\n", i, ecal_config.camera_topics[i].c_str());
  }

  // Main processing loop
  while (eCAL::Ok() && rclcpp::ok() && !shutdown_flag.load()) {
    // Process ROS2 callbacks
    rclcpp::spin_some(ros_node);

    // Sleep briefly to prevent busy waiting
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  PRINT_INFO("Shutting down eCAL VIO node...\n");
  ecal_node->shutdown();

  if (viz) {
    viz->visualize_final();
  }

  eCAL::Finalize();
  rclcpp::shutdown();

  return EXIT_SUCCESS;
}

void EcalVioNode::onImuMessage(const char* topic_name,
                              const struct eCAL::SReceiveCallbackData* data) {
  if (shutdown_flag.load() || !data || !data->buf || data->size == 0) {
    return;
  }

  try {
    kj::ArrayPtr<const kj::byte> bytes(reinterpret_cast<const kj::byte*>(data->buf), data->size);
    kj::ArrayInputStream stream(bytes);
    capnp::InputStreamMessageReader reader(stream);
    vkc::Imu::Reader imu_msg = reader.getRoot<vkc::Imu>();

    // Convert to OpenVINS IMU measurement
    ov_core::ImuData imu_measurement = convertImuMessage(imu_msg);

    // Feed to VIO system
    sys_->feed_measurement_imu(imu_measurement);

    // Visualize odometry at IMU rate for high frequency pose estimates
    if (viz) {
      viz->visualize_odometry(imu_measurement.timestamp);
    }

    imu_message_count_++;

    if (imu_message_count_ % 100 == 0) {
      PRINT_DEBUG("Received %zu IMU messages\n", imu_message_count_);
    }

  } catch (const std::exception& e) {
    PRINT_ERROR(RED "Error processing IMU message: %s\n" RESET, e.what());
  }
}

void EcalVioNode::onImageMessage(const char* topic_name,
                                const struct eCAL::SReceiveCallbackData* data, size_t cam_idx) {
  if (shutdown_flag.load() || !data || !data->buf || data->size == 0 || cam_idx >= buffered_images_.size()) {
    return;
  }

  try {
    kj::ArrayPtr<const kj::byte> bytes(reinterpret_cast<const kj::byte*>(data->buf), data->size);
    kj::ArrayInputStream stream(bytes);
    capnp::InputStreamMessageReader reader(stream);
    vkc::Image::Reader image_msg = reader.getRoot<vkc::Image>();

    // Convert to OpenCV image
    cv::Mat image = convertImageMessage(image_msg);
    if (image.empty()) {
      return;
    }

    // Extract timestamp
    auto header = image_msg.getHeader();
    double timestamp = (header.getStampMonotonic() + header.getClockOffset()) * 1e-9;

    // Buffer the image for this camera
    buffered_images_[cam_idx] = image.clone();
    buffered_timestamps_[cam_idx] = timestamp;

    image_counts_[cam_idx]++;

    // Create ROS2 image message for visualization
    sensor_msgs::msg::Image::SharedPtr ros_image = nullptr;
    if (viz) {
      std::string frame_id = "cam" + std::to_string(cam_idx);
      ros_image = convertToRosImage(image_msg, frame_id);
    }

    // Try to process camera images if we have synchronized data
    processCameraImages();

    // Call visualization callbacks with the image data
    if (viz && ros_image) {
      if (config_.camera_topics.size() == 1) {
        // Monocular case
        viz->callback_monocular(ros_image, static_cast<int>(cam_idx));
      } else if (config_.camera_topics.size() == 2 && cam_idx < 2) {
        // Store images for stereo callback
        static sensor_msgs::msg::Image::SharedPtr buffered_ros_images[2] = {nullptr, nullptr};
        buffered_ros_images[cam_idx] = ros_image;

        // If we have both stereo images, call stereo callback
        if (buffered_ros_images[0] && buffered_ros_images[1]) {
          // Check if timestamps are synchronized (within 10ms)
          double time_diff = std::abs(
            (buffered_ros_images[0]->header.stamp.sec + buffered_ros_images[0]->header.stamp.nanosec * 1e-9) -
            (buffered_ros_images[1]->header.stamp.sec + buffered_ros_images[1]->header.stamp.nanosec * 1e-9)
          );

          if (time_diff < 0.01) {
            viz->callback_stereo(buffered_ros_images[0], buffered_ros_images[1], 0, 1);
            buffered_ros_images[0] = nullptr;
            buffered_ros_images[1] = nullptr;
          }
        }
      }
    }

    if (image_counts_[cam_idx] % 30 == 0) {
      PRINT_DEBUG("Received %zu frames from camera %zu\n", image_counts_[cam_idx], cam_idx);
    }

  } catch (const std::exception& e) {
    PRINT_ERROR(RED "Error processing image message from camera %zu: %s\n" RESET, cam_idx, e.what());
  }
}

ov_core::ImuData EcalVioNode::convertImuMessage(const vkc::Imu::Reader& imu_msg) {
  ov_core::ImuData measurement;

  // Extract timestamp from header (convert nanoseconds to seconds)
  auto header = imu_msg.getHeader();
  measurement.timestamp = (header.getStampMonotonic() + header.getClockOffset()) * 1e-9;

  // Convert linear acceleration
  auto linear_acc = imu_msg.getLinearAcceleration();
  measurement.am(0) = linear_acc.getX();
  measurement.am(1) = linear_acc.getY();
  measurement.am(2) = linear_acc.getZ();

  // Convert angular velocity
  auto angular_vel = imu_msg.getAngularVelocity();
  measurement.wm(0) = angular_vel.getX();
  measurement.wm(1) = angular_vel.getY();
  measurement.wm(2) = angular_vel.getZ();

  return measurement;
}

cv::Mat EcalVioNode::convertImageMessage(const vkc::Image::Reader& image_msg) {
  // Get image properties
  uint32_t width = image_msg.getWidth();
  uint32_t height = image_msg.getHeight();
  auto encoding = image_msg.getEncoding();

  // Get image data
  auto data = image_msg.getData();

  // Convert image data to OpenCV Mat based on encoding
  cv::Mat cv_image;

  switch (encoding) {
    case vkc::Image::Encoding::MONO8: {
      cv_image = cv::Mat(height, width, CV_8UC1,
                         const_cast<void*>(static_cast<const void*>(data.begin())));
      break;
    }
    case vkc::Image::Encoding::BGR8: {
      cv_image = cv::Mat(height, width, CV_8UC3,
                         const_cast<void*>(static_cast<const void*>(data.begin())));
      // Convert BGR to grayscale for VIO
      cv::cvtColor(cv_image, cv_image, cv::COLOR_BGR2GRAY);
      break;
    }
    case vkc::Image::Encoding::MONO16: {
      cv_image = cv::Mat(height, width, CV_16UC1,
                         const_cast<void*>(static_cast<const void*>(data.begin())));
      // Convert to 8-bit if needed
      cv_image.convertTo(cv_image, CV_8UC1, 1.0/256.0);
      break;
    }
    case vkc::Image::Encoding::JPEG: {
      std::vector<uint8_t> compressed_data(data.begin(), data.end());
      cv_image = cv::imdecode(cv::Mat(compressed_data), cv::IMREAD_GRAYSCALE);
      if (cv_image.empty()) {
          PRINT_ERROR(RED "Failed to decode JPEG image\n" RESET);
          return cv::Mat();
      }
      if (!cv_image.isContinuous()) {
          cv_image = cv_image.clone();
      }
      break;
    }
    default:
      PRINT_ERROR(RED "Unsupported image encoding: %d\n" RESET, static_cast<int>(encoding));
      return cv::Mat();
  }

  // Clone the image to ensure we own the data
  return cv_image.clone();
}

sensor_msgs::msg::Image::SharedPtr EcalVioNode::convertToRosImage(const vkc::Image::Reader& image_msg, const std::string& frame_id) {
  // Get image properties
  uint32_t width = image_msg.getWidth();
  uint32_t height = image_msg.getHeight();
  auto encoding = image_msg.getEncoding();

  // Extract timestamp
  auto header = image_msg.getHeader();
  double timestamp = (header.getStampMonotonic() + header.getClockOffset()) * 1e-9;

  // Convert eCAL image to OpenCV Mat first
  cv::Mat cv_image = convertImageMessage(image_msg);
  if (cv_image.empty()) {
    return nullptr;
  }

  // Create ROS2 Image message
  auto ros_image = std::make_shared<sensor_msgs::msg::Image>();

  // Set header
  ros_image->header.stamp.sec = static_cast<int32_t>(timestamp);
  ros_image->header.stamp.nanosec = static_cast<uint32_t>((timestamp - ros_image->header.stamp.sec) * 1e9);
  ros_image->header.frame_id = frame_id;

  // Set image properties
  ros_image->height = cv_image.rows;
  ros_image->width = cv_image.cols;
  ros_image->encoding = "mono8";  // OpenVINS typically uses grayscale
  ros_image->is_bigendian = false;
  ros_image->step = cv_image.cols * cv_image.elemSize();

  // Copy image data
  size_t data_size = ros_image->step * ros_image->height;
  ros_image->data.resize(data_size);
  memcpy(ros_image->data.data(), cv_image.data, data_size);

  return ros_image;
}

void EcalVioNode::processCameraImages() {
  // Check if VIO system is valid and initialized
  if (!sys_) {
    PRINT_ERROR(RED "VIO system is null\n" RESET);
    return;
  }

  // If not initialized, try to initialize using camera data
  if (!sys_->initialized()) {
    // We need to attempt initialization, but first let's make sure we have valid data
    // Find the first valid camera data for initialization attempt
    for (size_t i = 0; i < buffered_images_.size(); ++i) {
      if (!buffered_images_[i].empty() && buffered_timestamps_[i] > 0) {
        // Validate image for initialization
        if (buffered_images_[i].rows < 10 || buffered_images_[i].cols < 10 ||
            buffered_images_[i].type() != CV_8UC1) {
          continue; // Skip invalid images
        }

        ov_core::CameraData init_data;
        init_data.timestamp = buffered_timestamps_[i];
        init_data.sensor_ids.push_back(static_cast<int>(i));
        init_data.images.push_back(buffered_images_[i].clone());
        init_data.masks.push_back(cv::Mat::zeros(buffered_images_[i].size(), CV_8UC1));

        try {
          // This will call track_image_and_update which handles initialization internally
          sys_->feed_measurement_camera(init_data);

          // Only visualize after successful VIO processing, and only if initialized
          if (viz && sys_->initialized()) {
            try {
              viz->visualize();
            } catch (const std::exception& viz_e) {
              PRINT_WARNING(YELLOW "Visualization exception during initialization: %s\n" RESET, viz_e.what());
            }
          }

          // Clear this buffer after initialization attempt
          buffered_timestamps_[i] = -1;
        } catch (const std::exception& e) {
          PRINT_ERROR(RED "Exception during VIO initialization: %s\n" RESET, e.what());
        }

        return; // Try one image at a time for initialization
      }
    }
    return; // No valid images for initialization
  }

  // Check if we have at least one camera with valid data
  bool has_valid_data = false;
  for (size_t i = 0; i < buffered_images_.size(); ++i) {
    if (!buffered_images_[i].empty() && buffered_timestamps_[i] > 0) {
      has_valid_data = true;
      break;
    }
  }

  if (!has_valid_data) {
    return;
  }

  // For monocular case, process immediately
  if (buffered_images_.size() == 1) {
    if (!buffered_images_[0].empty() && buffered_timestamps_[0] > 0) {
      // Validate image dimensions
      if (buffered_images_[0].rows < 10 || buffered_images_[0].cols < 10) {
        PRINT_ERROR(RED "Invalid image dimensions: %dx%d\n" RESET, buffered_images_[0].cols, buffered_images_[0].rows);
        buffered_timestamps_[0] = -1;
        return;
      }

      // Validate image data
      if (buffered_images_[0].type() != CV_8UC1) {
        PRINT_ERROR(RED "Invalid image type: %d (expected CV_8UC1=%d)\n" RESET, buffered_images_[0].type(), CV_8UC1);
        buffered_timestamps_[0] = -1;
        return;
      }

      ov_core::CameraData cam_data;
      cam_data.timestamp = buffered_timestamps_[0];
      cam_data.sensor_ids.push_back(0);
      cam_data.images.push_back(buffered_images_[0].clone());
      cam_data.masks.push_back(cv::Mat::zeros(buffered_images_[0].size(), CV_8UC1));

      // Additional validation
      if (cam_data.sensor_ids.size() != cam_data.images.size() ||
          cam_data.images.size() != cam_data.masks.size()) {
        PRINT_ERROR(RED "Camera data size mismatch: ids=%zu, images=%zu, masks=%zu\n" RESET,
                   cam_data.sensor_ids.size(), cam_data.images.size(), cam_data.masks.size());
        buffered_timestamps_[0] = -1;
        return;
      }

      try {
        // Feed to VIO system
        sys_->feed_measurement_camera(cam_data);

        // Visualize the current state and features
        if (viz) {
          viz->visualize();
        }
      } catch (const std::exception& e) {
        PRINT_ERROR(RED "Exception in VIO processing: %s\n" RESET, e.what());
      }

      // Clear buffer
      buffered_timestamps_[0] = -1;
    }
    return;
  }

  // For multi-camera case, check synchronization
  if (allCamerasSynchronized()) {
    // Use the first camera's timestamp as reference
    double reference_timestamp = buffered_timestamps_[0];

    ov_core::CameraData cam_data;
    cam_data.timestamp = reference_timestamp;

    // Add all synchronized cameras with validation
    for (size_t i = 0; i < buffered_images_.size(); ++i) {
      if (!buffered_images_[i].empty() && buffered_timestamps_[i] > 0) {
        // Validate image dimensions and type
        if (buffered_images_[i].rows < 10 || buffered_images_[i].cols < 10) {
          PRINT_WARNING(YELLOW "Skipping camera %zu: invalid dimensions %dx%d\n" RESET,
                       i, buffered_images_[i].cols, buffered_images_[i].rows);
          continue;
        }

        if (buffered_images_[i].type() != CV_8UC1) {
          PRINT_WARNING(YELLOW "Skipping camera %zu: invalid type %d (expected CV_8UC1=%d)\n" RESET,
                       i, buffered_images_[i].type(), CV_8UC1);
          continue;
        }

        cam_data.sensor_ids.push_back(static_cast<int>(i));
        cam_data.images.push_back(buffered_images_[i].clone());
        cam_data.masks.push_back(cv::Mat::zeros(buffered_images_[i].size(), CV_8UC1));
      }
    }

    // Feed to VIO system if we have at least one valid camera
    if (!cam_data.images.empty()) {
      // Final validation
      if (cam_data.sensor_ids.size() != cam_data.images.size() ||
          cam_data.images.size() != cam_data.masks.size()) {
        PRINT_ERROR(RED "Multi-camera data size mismatch: ids=%zu, images=%zu, masks=%zu\n" RESET,
                   cam_data.sensor_ids.size(), cam_data.images.size(), cam_data.masks.size());
      } else {
        try {
          sys_->feed_measurement_camera(cam_data);

          // Visualize the current state and features
          if (viz) {
            viz->visualize();
          }
        } catch (const std::exception& e) {
          PRINT_ERROR(RED "Exception in multi-camera VIO processing: %s\n" RESET, e.what());
        }
      }

      // Clear all buffers
      for (size_t i = 0; i < buffered_timestamps_.size(); ++i) {
        buffered_timestamps_[i] = -1;
      }
    }
  }
}

bool EcalVioNode::allCamerasSynchronized(double sync_threshold) {
  // Find the first valid timestamp as reference
  double reference_timestamp = -1;
  for (size_t i = 0; i < buffered_timestamps_.size(); ++i) {
    if (buffered_timestamps_[i] > 0 && !buffered_images_[i].empty()) {
      reference_timestamp = buffered_timestamps_[i];
      break;
    }
  }

  if (reference_timestamp < 0) {
    return false; // No valid timestamps
  }

  // Check if all cameras have data within sync threshold
  size_t synchronized_cameras = 0;
  for (size_t i = 0; i < buffered_timestamps_.size(); ++i) {
    if (buffered_timestamps_[i] > 0 && !buffered_images_[i].empty() &&
        std::abs(buffered_timestamps_[i] - reference_timestamp) < sync_threshold) {
      synchronized_cameras++;
    }
  }

  // For stereo, we need at least 2 cameras synchronized
  // For mono, we need at least 1 camera
  return (buffered_images_.size() == 1 && synchronized_cameras >= 1) ||
         (buffered_images_.size() > 1 && synchronized_cameras >= 2);
}