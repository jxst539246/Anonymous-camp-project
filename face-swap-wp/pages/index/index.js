//index.js
//获取应用实例
const app = getApp()

import WeCropper from '../../we-cropper/we-cropper.js'

const device = wx.getSystemInfoSync() // 获取设备信息
const width = device.windowWidth - 60 // 示例为一个与屏幕等宽的正方形裁剪框
const height = width

Page({
  data: {
    userInfo: {},
    hasUserInfo: false,
    canIUse: wx.canIUse('button.open-type.getUserInfo'),
    imgReady: false,
    imgurl: '/assets/add.jpg',
    style: '0',
    msg: '',
    cropperOpt: {
      id: 'cropper', // 用于手势操作的canvas组件标识符
      targetId: 'targetCropper', // 用于用于生成截图的canvas组件标识符
      pixelRatio: device.pixelRatio, // 传入设备像素比
      width,  // 画布宽度
      height, // 画布高度
      scale: 2.5, // 最大缩放倍数
      zoom: 8, // 缩放系数
      cut: {
        x: 20, // 裁剪框x轴起点
        y: 20, // 裁剪框y轴期起点
        width: device.windowWidth-100, // 裁剪框宽度
        height: device.windowWidth - 100 // 裁剪框高度
      }
    }
  },
  onLoad: function () {
    if (app.globalData.userInfo) {
      this.setData({
        userInfo: app.globalData.userInfo,
        hasUserInfo: true
      })
    } else if (this.data.canIUse){
      // 由于 getUserInfo 是网络请求，可能会在 Page.onLoad 之后才返回
      // 所以此处加入 callback 以防止这种情况
      app.userInfoReadyCallback = res => {
        this.setData({
          userInfo: res.userInfo,
          hasUserInfo: true
        })
      }
    } else {
      // 在没有 open-type=getUserInfo 版本的兼容处理
      wx.getUserInfo({
        success: res => {
          app.globalData.userInfo = res.userInfo
          this.setData({
            userInfo: res.userInfo,
            hasUserInfo: true
          })
        }
      })
    }
    const { cropperOpt } = this.data

    this.cropper = new WeCropper(cropperOpt)
      .on('ready', (ctx) => {
        console.log(`wecropper is ready for work!`)
      })
      .on('beforeImageLoad', (ctx) => {
        wx.showToast({
          title: '上传中',
          icon: 'loading',
          duration: 20000
        })
      })
      .on('imageLoad', (ctx) => {
        wx.hideToast()
      })
  },
  getUserInfo: function(e) {
    console.log(e)
    app.globalData.userInfo = e.detail.userInfo
    this.setData({
      userInfo: e.detail.userInfo,
      hasUserInfo: true
    })
  },
  setStyle(e) {
    this.setData({ style: e.target.id});
  },
  uploadTap() {
    const self = this

    wx.chooseImage({
      count: 1, // 默认9
      sizeType: ['compressed'], // 可以指定是原图还是压缩图，默认二者都有
      sourceType: ['album', 'camera'], // 可以指定来源是相册还是相机，默认二者都有
      success(res) {
        const src = res.tempFilePaths[0]
        self.cropper.pushOrign(src)
        self.setData({imgReady: true})
      }
    })

  },
  getCropperImage() {
    const self = this
    this.cropper.getCropperBase64((base64) => {
      // tempFilePath 为裁剪后的图片临时路径
      if (base64) {
        self.setData({ imgReady: false })
        console.log({
          img: base64,
          style: self.data.style,
          timestamp: new Date().getTime()
        })
        wx.showToast({
          title: '上传中',
          icon: 'loading',
          duration: 20000
        })
        wx.request({
          url: 'https://miao.menghj.com/transformer/trans/', 
          data: {
            image: base64,
            style: self.data.style,
            timestamp: new Date().getTime()
          },
          method: 'POST',
          header: {
            'content-type': 'application/json' // 默认值
          },
          success(res) {
            wx.hideToast()
            console.log(res.data)
            wx.previewImage({
              current: '',
              urls: ["https://miao.menghj.com/images/" + res.data.file_name]
            })
          },
          fail(res) {
            wx.hideToast()
            self.setData({msg: JSON.stringify(res)})
          }
        })

        
      } else {
        console.log('获取图片地址失败，请稍后重试')
      }
    })
  },
  touchStart(e) {
    this.cropper.touchStart(e)
  },
  touchMove(e) {
    this.cropper.touchMove(e)
  },
  touchEnd(e) {
    this.cropper.touchEnd(e)
  }
})
