"""
Django settings for server project.

Generated by 'django-admin startproject' using Django 2.0.1.

For more information on this file, see
https://docs.djangoproject.com/en/2.0/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/2.0/ref/settings/
"""

import os

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/2.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'fr)5j30-=kc%(69d_4!#pytr9p%@2#_)s@$e7pvx^okprphjyb'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'transformer'
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    # 'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'request_logging.middleware.LoggingMiddleware',
]

ROOT_URLCONF = 'server.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'server.wsgi.application'

# Database
# https://docs.djangoproject.com/en/2.0/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}

# Password validation
# https://docs.djangoproject.com/en/2.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
# https://docs.djangoproject.com/en/2.0/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True

LOGGING = {
    'version': 1,  # 日志版本
    'disable_existing_loggers': False,  # True：disable原有日志相关配置
    'formatters': {  # 日志格式
        'verbose': {  # 详细格式
            'format': '%(levelname)s %(asctime)s %(module)s %(process)d %(thread)d %(message)s'
        },
        'simple': {  # 简单格式
            'format': '%(levelname)s %(message)s'
        }
    },
    'filters': {  # 日志过滤器
        'require_debug_true': {  # 是否支持DEBUG级别日志过滤
            '()': 'django.utils.log.RequireDebugTrue',
        },
    },
    'handlers': {  # 日志handlers
        'null': {
            'level': 'DEBUG',
            'class': 'logging.NullHandler',
        },
        'file': {  # 文件handler
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': 'logs/verbose.log',
            # 'when': config.get('logging', 'when'),
            # 'interval': config.getint('logging', 'interval'),
            # 'backupCount': config.getint('logging', 'backup_count'),
            'formatter': 'verbose'
        },
        'console': {  # 控制器handler，INFO级别以上的日志都要Simple格式输出到控制台
            'level': 'INFO',
            'filters': ['require_debug_true'],
            'class': 'logging.StreamHandler',
            'formatter': 'simple'
        },
        'debug-info': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': 'logs/info.log',
            'formatter': 'verbose',
        }
    },
    'loggers': {
        # 'django': {
        #     'handlers': ['console'],
        #     'propagate': True,
        # },
        # 'django.server': {
        #     'handlers': ['console'],
        #     'level': 'INFO',
        #     'propagate': False,
        # },
        'django.request': {
            'handlers': ['file'],
            'level': 'DEBUG',  # change debug level as appropiate
            'propagate': False,
        },
        'debug-info-log': {
            'handlers': ['debug-info'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.0/howto/static-files/

STATIC_URL = '/static/'
STATIC_ROOT = '/home/menruimr/Anonymous-camp-project/server/server/static'
