#include <csignal>
#include <memory>
#include <iostream>
#include <atomic>
#include <thread>
#include <chrono>
#include <deque>
#include <mutex>
#include <opencv2/opencv.hpp>
#include "types/IMU.h"
#include "state/State.h"

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
#include "odometry3d.capnp.h"

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
  size_t max_buffer_size = 2; // Maximum number of images to buffer per stream
  double sync_tolerance_sec = 0.02; // Maximum time difference for synchronization
};

// Structure to hold buffered image data
struct BufferedImage {
  cv::Mat image;
  uint64_t timestamp;
  sensor_msgs::msg::Image::SharedPtr ros_image;

  BufferedImage(const cv::Mat& img, uint64_t ts) : image(img.clone()), timestamp(ts), ros_image(nullptr) {}
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
    image_buffers_.resize(config_.camera_topics.size());
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

    // Create output publisher
    output_publisher_ = std::make_unique<eCAL::CPublisher>("/vio", "capnp:Odometry3D", "OpenVINS VIO output");
    if (!output_publisher_->IsCreated()) {
      PRINT_ERROR(RED "Failed to create output publisher for topic: /vio/odometry\n" RESET);
      return false;
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
  void publishVioOutput();

  // Synchronization helper functions
  uint64_t findSynchronizedImages(std::vector<size_t>& sync_indices);
  void removeOldestImages();
  void removeSynchronizedImages(const std::vector<size_t>& sync_indices);

  EcalConfig config_;
  std::shared_ptr<VioManager> sys_;

  std::unique_ptr<eCAL::CSubscriber> imu_subscriber_;
  std::vector<std::unique_ptr<eCAL::CSubscriber>> camera_subscribers_;
  std::unique_ptr<eCAL::CPublisher> output_publisher_;

  std::vector<std::deque<BufferedImage>> image_buffers_;
  std::mutex buffer_mutex_;
  sensor_msgs::msg::Image::SharedPtr buffered_ros_images_[2];

  size_t imu_message_count_ = 0;
  std::vector<size_t> image_counts_;
  uint32_t seq_ = 0;
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
  if (shutdown_flag.load() || !data || !data->buf || data->size == 0 || cam_idx >= image_buffers_.size()) {
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
    uint64_t timestamp = (header.getStampMonotonic() + header.getClockOffset());

    // Buffer the image for this camera
    {
      std::lock_guard<std::mutex> lock(buffer_mutex_);

      // Add new image to buffer
      image_buffers_[cam_idx].emplace_back(image, timestamp);

      // Create ROS2 image message for visualization (only for first two cameras)
      if (viz && cam_idx < 2) {
        std::string frame_id = "cam" + std::to_string(cam_idx);
        image_buffers_[cam_idx].back().ros_image = convertToRosImage(image_msg, frame_id);
      }

      // Maintain buffer size limit
      while (image_buffers_[cam_idx].size() > config_.max_buffer_size) {
        image_buffers_[cam_idx].pop_front();
        PRINT_WARNING(YELLOW "Buffer overflow for camera %zu, dropping oldest image\n" RESET, cam_idx);
      }
    }

    image_counts_[cam_idx]++;

    // Try to process camera images if we have synchronized data
    processCameraImages();

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
  measurement.timestamp = static_cast<double>(header.getStampMonotonic() + header.getClockOffset())/1e9;

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



void EcalVioNode::processCameraImages() {
  std::lock_guard<std::mutex> lock(buffer_mutex_);

  if (config_.camera_topics.size() == 1) {
    // Single camera case - process oldest available image
    if (image_buffers_[0].empty()) {
      return;
    }

    const BufferedImage& buffered_img = image_buffers_[0].front();

    ov_core::CameraData cam_data;
    cam_data.timestamp = static_cast<double>(buffered_img.timestamp) / 1e9;
    cam_data.sensor_ids.push_back(0);
    cam_data.images.push_back(buffered_img.image.clone());
    cam_data.masks.push_back(cv::Mat::zeros(buffered_img.image.size(), CV_8UC1));

    // Store ROS image for visualization
    if (buffered_img.ros_image) {
      buffered_ros_images_[0] = buffered_img.ros_image;
    }

    try {
      // Feed to VIO system
      sys_->feed_measurement_camera(cam_data);
    } catch (const std::exception& e) {
      PRINT_ERROR(RED "Exception in VIO processing: %s\n" RESET, e.what());
    }

    // Remove processed image
    image_buffers_[0].pop_front();
  }
  // Multi-camera case - find synchronized images
  else {
    // Check if all cameras have at least one image
    for (size_t i = 0; i < image_buffers_.size(); ++i) {
      if (image_buffers_[i].empty()) {
        return; // Wait for all cameras to have data
      }
    }

    // Find synchronized images using oldest timestamp from each camera
    std::vector<size_t> sync_indices(image_buffers_.size(), 0);
    uint64_t target_timestamp = findSynchronizedImages(sync_indices);

    if (target_timestamp == 0) {
      // No synchronized images found, remove oldest images to make progress
      removeOldestImages();
      return;
    }

    // Create camera data with synchronized images
    ov_core::CameraData cam_data;
    cam_data.timestamp = static_cast<double>(target_timestamp) / 1e9;

    for (size_t idx = 0; idx < image_buffers_.size(); ++idx) {
      const BufferedImage& sync_img = image_buffers_[idx][sync_indices[idx]];
      cam_data.sensor_ids.push_back(static_cast<int>(idx));
      cam_data.images.push_back(sync_img.image.clone());
      cam_data.masks.push_back(cv::Mat::zeros(sync_img.image.size(), CV_8UC1));

      // Store ROS images for visualization (first two cameras)
      if (idx < 2 && sync_img.ros_image) {
        buffered_ros_images_[idx] = sync_img.ros_image;
      }
    }

    try {
      sys_->feed_measurement_camera(cam_data);
    } catch (const std::exception& e) {
      PRINT_ERROR(RED "Exception in multi-camera VIO processing: %s\n" RESET, e.what());
    }

    // Remove processed images
    removeSynchronizedImages(sync_indices);
  }

  publishVioOutput();

  // Visualize the current state and features
  if (viz) {
    viz->visualize(); 
    if (config_.camera_topics.size() == 1) {
      // Monocular case
      viz->callback_monocular(buffered_ros_images_[0], 0);
    } else {
      // Store images for stereo callback

      viz->callback_stereo(buffered_ros_images_[0], buffered_ros_images_[1], 0, 1);
    } 
  }
}


void EcalVioNode::publishVioOutput() {
  if (!output_publisher_ || !sys_ || !sys_->initialized()) {
    return;
  }

  try {
    // Get current state from OpenVINS
    std::shared_ptr<ov_msckf::State> state = sys_->get_state();
    if (!state) {
      return;
    }

    // Get IMU state
    std::shared_ptr<ov_type::IMU> imu_state = state->_imu;
    if (!imu_state) {
      return;
    }

    // Extract position (IMU to global frame)
    Eigen::Vector3d pos_ItoG = imu_state->pos();

    // Extract quaternion 
    // TODO: check frame here
    Eigen::Matrix<double, 4, 1> quat = imu_state->quat();

    // Extract velocity (in global frame)
    Eigen::Vector3d vel_IinG = imu_state->vel();

    capnp::MallocMessageBuilder message;
    vkc::Odometry3d::Builder odometry = message.initRoot<vkc::Odometry3d>();

    // Set header information
    auto header = odometry.initHeader();
    header.setSeq(seq_++);
    header.setStampMonotonic(state->_timestamp);

    // Set position and orientation
    auto pose = odometry.initPose();
    auto position = pose.initPosition();
    position.setX(pos_ItoG.x());
    position.setY(pos_ItoG.y());
    position.setZ(pos_ItoG.z());

    // Set orientation quaternion (IMU to global frame)
    auto orientation = pose.initOrientation();
    orientation.setW(quat(3));
    orientation.setX(quat(0));
    orientation.setY(quat(1));
    orientation.setZ(quat(2));

    // Set linear velocity
    auto twist = odometry.initTwist();
    auto linear = twist.initLinear();
    linear.setX(vel_IinG.x());
    linear.setY(vel_IinG.y());
    linear.setZ(vel_IinG.z());

    // Angular velocity not directly available from state, set to zero for now
    // auto angular = twist.initAngular();
    // angular.setX(0.0);
    // angular.setY(0.0);
    // angular.setZ(0.0);

    // Serialize and publish
    kj::Array<capnp::word> words = capnp::messageToFlatArray(message);
    kj::ArrayPtr<const char> array(reinterpret_cast<const char*>(words.begin()),
                                   words.size() * sizeof(capnp::word));
    output_publisher_->Send(array.begin(), array.size());

  } catch (const std::exception& e) {
    PRINT_ERROR(RED "Exception in VIO output publishing: %s\n" RESET, e.what());
  }
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

uint64_t EcalVioNode::findSynchronizedImages(std::vector<size_t>& sync_indices) {
  // Find the oldest timestamp among all first images in each buffer
  uint64_t oldest_timestamp = UINT64_MAX;
  for (size_t cam = 0; cam < image_buffers_.size(); ++cam) {
    if (!image_buffers_[cam].empty()) {
      oldest_timestamp = std::min(oldest_timestamp, image_buffers_[cam][0].timestamp);
    }
  }

  if (oldest_timestamp == UINT64_MAX) {
    return 0; // No images available
  }

  // For each camera, find the image closest to the oldest timestamp
  double sync_tolerance_ns = config_.sync_tolerance_sec * 1e9;
  bool all_synchronized = true;

  for (size_t cam = 0; cam < image_buffers_.size(); ++cam) {
    bool found_sync = false;

    for (size_t i = 0; i < image_buffers_[cam].size(); ++i) {
      double time_diff = std::abs(static_cast<double>(image_buffers_[cam][i].timestamp) -
                                  static_cast<double>(oldest_timestamp));

      if (time_diff <= sync_tolerance_ns) {
        sync_indices[cam] = i;
        found_sync = true;
        break;
      }
    }

    if (!found_sync) {
      all_synchronized = false;
      break;
    }
  }

  if (!all_synchronized) {
    PRINT_WARNING(YELLOW "Could not find synchronized images within tolerance of %.3f ms\n" RESET, 
                config_.sync_tolerance_sec * 1000.0);
    return 0;
  }

  return oldest_timestamp;
}

void EcalVioNode::removeOldestImages() {
  // Find camera with oldest image and remove it
  uint64_t oldest_timestamp = UINT64_MAX;
  size_t oldest_cam = 0;

  for (size_t cam = 0; cam < image_buffers_.size(); ++cam) {
    if (!image_buffers_[cam].empty() && image_buffers_[cam][0].timestamp < oldest_timestamp) {
      oldest_timestamp = image_buffers_[cam][0].timestamp;
      oldest_cam = cam;
    }
  }

  if (oldest_timestamp != UINT64_MAX) {
    image_buffers_[oldest_cam].pop_front();
    PRINT_DEBUG("Removed oldest unsynchronized image from camera %zu\n", oldest_cam);
  }
}

void EcalVioNode::removeSynchronizedImages(const std::vector<size_t>& sync_indices) {
  // Remove synchronized images from each buffer
  // Note: Remove in reverse order of indices to avoid index shifting issues
  for (size_t cam = 0; cam < image_buffers_.size(); ++cam) {
    if (sync_indices[cam] < image_buffers_[cam].size()) {
      // Remove elements from front up to and including the synchronized index
      for (size_t i = 0; i <= sync_indices[cam]; ++i) {
        if (!image_buffers_[cam].empty()) {
          image_buffers_[cam].pop_front();
        }
      }
    }
  }
}