// GENERATED CODE - DO NOT MODIFY BY HAND
// coverage:ignore-file
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'location_model.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

// dart format off
T _$identity<T>(T value) => value;

/// @nodoc
mixin _$Location {

@JsonKey(name: 'y', fromJson: _stringToDouble) double get latitude;@JsonKey(name: 'x', fromJson: _stringToDouble) double get longitude;@JsonKey(name: 'place_name') String get placeName;@JsonKey(name: 'address_name') String get addressName;@JsonKey(name: 'road_address_name') String get roadAddressName;@JsonKey(name: 'category_name') String get categoryName;
/// Create a copy of Location
/// with the given fields replaced by the non-null parameter values.
@JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
$LocationCopyWith<Location> get copyWith => _$LocationCopyWithImpl<Location>(this as Location, _$identity);

  /// Serializes this Location to a JSON map.
  Map<String, dynamic> toJson();


@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is Location&&(identical(other.latitude, latitude) || other.latitude == latitude)&&(identical(other.longitude, longitude) || other.longitude == longitude)&&(identical(other.placeName, placeName) || other.placeName == placeName)&&(identical(other.addressName, addressName) || other.addressName == addressName)&&(identical(other.roadAddressName, roadAddressName) || other.roadAddressName == roadAddressName)&&(identical(other.categoryName, categoryName) || other.categoryName == categoryName));
}

@JsonKey(includeFromJson: false, includeToJson: false)
@override
int get hashCode => Object.hash(runtimeType,latitude,longitude,placeName,addressName,roadAddressName,categoryName);

@override
String toString() {
  return 'Location(latitude: $latitude, longitude: $longitude, placeName: $placeName, addressName: $addressName, roadAddressName: $roadAddressName, categoryName: $categoryName)';
}


}

/// @nodoc
abstract mixin class $LocationCopyWith<$Res>  {
  factory $LocationCopyWith(Location value, $Res Function(Location) _then) = _$LocationCopyWithImpl;
@useResult
$Res call({
@JsonKey(name: 'y', fromJson: _stringToDouble) double latitude,@JsonKey(name: 'x', fromJson: _stringToDouble) double longitude,@JsonKey(name: 'place_name') String placeName,@JsonKey(name: 'address_name') String addressName,@JsonKey(name: 'road_address_name') String roadAddressName,@JsonKey(name: 'category_name') String categoryName
});




}
/// @nodoc
class _$LocationCopyWithImpl<$Res>
    implements $LocationCopyWith<$Res> {
  _$LocationCopyWithImpl(this._self, this._then);

  final Location _self;
  final $Res Function(Location) _then;

/// Create a copy of Location
/// with the given fields replaced by the non-null parameter values.
@pragma('vm:prefer-inline') @override $Res call({Object? latitude = null,Object? longitude = null,Object? placeName = null,Object? addressName = null,Object? roadAddressName = null,Object? categoryName = null,}) {
  return _then(_self.copyWith(
latitude: null == latitude ? _self.latitude : latitude // ignore: cast_nullable_to_non_nullable
as double,longitude: null == longitude ? _self.longitude : longitude // ignore: cast_nullable_to_non_nullable
as double,placeName: null == placeName ? _self.placeName : placeName // ignore: cast_nullable_to_non_nullable
as String,addressName: null == addressName ? _self.addressName : addressName // ignore: cast_nullable_to_non_nullable
as String,roadAddressName: null == roadAddressName ? _self.roadAddressName : roadAddressName // ignore: cast_nullable_to_non_nullable
as String,categoryName: null == categoryName ? _self.categoryName : categoryName // ignore: cast_nullable_to_non_nullable
as String,
  ));
}

}


/// Adds pattern-matching-related methods to [Location].
extension LocationPatterns on Location {
/// A variant of `map` that fallback to returning `orElse`.
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case final Subclass value:
///     return ...;
///   case _:
///     return orElse();
/// }
/// ```

@optionalTypeArgs TResult maybeMap<TResult extends Object?>(TResult Function( _Location value)?  $default,{required TResult orElse(),}){
final _that = this;
switch (_that) {
case _Location() when $default != null:
return $default(_that);case _:
  return orElse();

}
}
/// A `switch`-like method, using callbacks.
///
/// Callbacks receives the raw object, upcasted.
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case final Subclass value:
///     return ...;
///   case final Subclass2 value:
///     return ...;
/// }
/// ```

@optionalTypeArgs TResult map<TResult extends Object?>(TResult Function( _Location value)  $default,){
final _that = this;
switch (_that) {
case _Location():
return $default(_that);case _:
  throw StateError('Unexpected subclass');

}
}
/// A variant of `map` that fallback to returning `null`.
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case final Subclass value:
///     return ...;
///   case _:
///     return null;
/// }
/// ```

@optionalTypeArgs TResult? mapOrNull<TResult extends Object?>(TResult? Function( _Location value)?  $default,){
final _that = this;
switch (_that) {
case _Location() when $default != null:
return $default(_that);case _:
  return null;

}
}
/// A variant of `when` that fallback to an `orElse` callback.
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case Subclass(:final field):
///     return ...;
///   case _:
///     return orElse();
/// }
/// ```

@optionalTypeArgs TResult maybeWhen<TResult extends Object?>(TResult Function(@JsonKey(name: 'y', fromJson: _stringToDouble)  double latitude, @JsonKey(name: 'x', fromJson: _stringToDouble)  double longitude, @JsonKey(name: 'place_name')  String placeName, @JsonKey(name: 'address_name')  String addressName, @JsonKey(name: 'road_address_name')  String roadAddressName, @JsonKey(name: 'category_name')  String categoryName)?  $default,{required TResult orElse(),}) {final _that = this;
switch (_that) {
case _Location() when $default != null:
return $default(_that.latitude,_that.longitude,_that.placeName,_that.addressName,_that.roadAddressName,_that.categoryName);case _:
  return orElse();

}
}
/// A `switch`-like method, using callbacks.
///
/// As opposed to `map`, this offers destructuring.
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case Subclass(:final field):
///     return ...;
///   case Subclass2(:final field2):
///     return ...;
/// }
/// ```

@optionalTypeArgs TResult when<TResult extends Object?>(TResult Function(@JsonKey(name: 'y', fromJson: _stringToDouble)  double latitude, @JsonKey(name: 'x', fromJson: _stringToDouble)  double longitude, @JsonKey(name: 'place_name')  String placeName, @JsonKey(name: 'address_name')  String addressName, @JsonKey(name: 'road_address_name')  String roadAddressName, @JsonKey(name: 'category_name')  String categoryName)  $default,) {final _that = this;
switch (_that) {
case _Location():
return $default(_that.latitude,_that.longitude,_that.placeName,_that.addressName,_that.roadAddressName,_that.categoryName);case _:
  throw StateError('Unexpected subclass');

}
}
/// A variant of `when` that fallback to returning `null`
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case Subclass(:final field):
///     return ...;
///   case _:
///     return null;
/// }
/// ```

@optionalTypeArgs TResult? whenOrNull<TResult extends Object?>(TResult? Function(@JsonKey(name: 'y', fromJson: _stringToDouble)  double latitude, @JsonKey(name: 'x', fromJson: _stringToDouble)  double longitude, @JsonKey(name: 'place_name')  String placeName, @JsonKey(name: 'address_name')  String addressName, @JsonKey(name: 'road_address_name')  String roadAddressName, @JsonKey(name: 'category_name')  String categoryName)?  $default,) {final _that = this;
switch (_that) {
case _Location() when $default != null:
return $default(_that.latitude,_that.longitude,_that.placeName,_that.addressName,_that.roadAddressName,_that.categoryName);case _:
  return null;

}
}

}

/// @nodoc
@JsonSerializable()

class _Location implements Location {
  const _Location({@JsonKey(name: 'y', fromJson: _stringToDouble) required this.latitude, @JsonKey(name: 'x', fromJson: _stringToDouble) required this.longitude, @JsonKey(name: 'place_name') required this.placeName, @JsonKey(name: 'address_name') required this.addressName, @JsonKey(name: 'road_address_name') this.roadAddressName = '', @JsonKey(name: 'category_name') this.categoryName = ''});
  factory _Location.fromJson(Map<String, dynamic> json) => _$LocationFromJson(json);

@override@JsonKey(name: 'y', fromJson: _stringToDouble) final  double latitude;
@override@JsonKey(name: 'x', fromJson: _stringToDouble) final  double longitude;
@override@JsonKey(name: 'place_name') final  String placeName;
@override@JsonKey(name: 'address_name') final  String addressName;
@override@JsonKey(name: 'road_address_name') final  String roadAddressName;
@override@JsonKey(name: 'category_name') final  String categoryName;

/// Create a copy of Location
/// with the given fields replaced by the non-null parameter values.
@override @JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
_$LocationCopyWith<_Location> get copyWith => __$LocationCopyWithImpl<_Location>(this, _$identity);

@override
Map<String, dynamic> toJson() {
  return _$LocationToJson(this, );
}

@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is _Location&&(identical(other.latitude, latitude) || other.latitude == latitude)&&(identical(other.longitude, longitude) || other.longitude == longitude)&&(identical(other.placeName, placeName) || other.placeName == placeName)&&(identical(other.addressName, addressName) || other.addressName == addressName)&&(identical(other.roadAddressName, roadAddressName) || other.roadAddressName == roadAddressName)&&(identical(other.categoryName, categoryName) || other.categoryName == categoryName));
}

@JsonKey(includeFromJson: false, includeToJson: false)
@override
int get hashCode => Object.hash(runtimeType,latitude,longitude,placeName,addressName,roadAddressName,categoryName);

@override
String toString() {
  return 'Location(latitude: $latitude, longitude: $longitude, placeName: $placeName, addressName: $addressName, roadAddressName: $roadAddressName, categoryName: $categoryName)';
}


}

/// @nodoc
abstract mixin class _$LocationCopyWith<$Res> implements $LocationCopyWith<$Res> {
  factory _$LocationCopyWith(_Location value, $Res Function(_Location) _then) = __$LocationCopyWithImpl;
@override @useResult
$Res call({
@JsonKey(name: 'y', fromJson: _stringToDouble) double latitude,@JsonKey(name: 'x', fromJson: _stringToDouble) double longitude,@JsonKey(name: 'place_name') String placeName,@JsonKey(name: 'address_name') String addressName,@JsonKey(name: 'road_address_name') String roadAddressName,@JsonKey(name: 'category_name') String categoryName
});




}
/// @nodoc
class __$LocationCopyWithImpl<$Res>
    implements _$LocationCopyWith<$Res> {
  __$LocationCopyWithImpl(this._self, this._then);

  final _Location _self;
  final $Res Function(_Location) _then;

/// Create a copy of Location
/// with the given fields replaced by the non-null parameter values.
@override @pragma('vm:prefer-inline') $Res call({Object? latitude = null,Object? longitude = null,Object? placeName = null,Object? addressName = null,Object? roadAddressName = null,Object? categoryName = null,}) {
  return _then(_Location(
latitude: null == latitude ? _self.latitude : latitude // ignore: cast_nullable_to_non_nullable
as double,longitude: null == longitude ? _self.longitude : longitude // ignore: cast_nullable_to_non_nullable
as double,placeName: null == placeName ? _self.placeName : placeName // ignore: cast_nullable_to_non_nullable
as String,addressName: null == addressName ? _self.addressName : addressName // ignore: cast_nullable_to_non_nullable
as String,roadAddressName: null == roadAddressName ? _self.roadAddressName : roadAddressName // ignore: cast_nullable_to_non_nullable
as String,categoryName: null == categoryName ? _self.categoryName : categoryName // ignore: cast_nullable_to_non_nullable
as String,
  ));
}


}

// dart format on
