from django.contrib import admin
from users.models import CustomUser
from django.contrib.auth.admin import UserAdmin

# Register your models here.
# admin.site.register(CustomUser)
# admin.site.register(Profile)

@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    search_fields = ['username', 'first_name', 'last_name',]
    list_display = ('id', 'username', 'first_name', 'last_name',)
    filter_horizontal = ('groups', 'user_permissions')
    fieldsets = (
        ('User', {'fields': ('username', 'password')}),
        ('Personal Informations', {'fields': (
            'first_name',
            'last_name',
            'email',
        )}),
        ('Permissions', {'fields': (
            'is_active',
            'is_staff',
            'is_superuser',
            'groups',
            'user_permissions'
        )}),
    )